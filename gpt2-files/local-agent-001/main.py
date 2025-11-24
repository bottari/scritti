from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pyttsx3
import sounddevice as sd
import vosk
import queue
import json
import re
import threading
import time
import sys
import msvcrt  # Windows-specific keyboard input

# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = r"models\full_merged_gpt2-finetuned-poetry-mercury-04--copy-attempt"
VOSK_MODEL_PATH = r"C:\Users\micha\Desktop\projects\local-agent-001\vosk-model-en-us-0.22"

SAMPLERATE = 16000
BLOCKSIZE = 4000
SILENCE_TIMEOUT = 2.5  # seconds of silence before finalizing speech
MIN_SPEECH_LENGTH = 3  # minimum words before considering finalization
MIN_CHAR_LENGTH = 10  # minimum character count to avoid stray words
IGNORE_WORDS = {"the", "a", "an", "uh", "um", "oh", "ah"}  # filter out stray filler words

# TTS Voice Configuration
TTS_RATE = 190  # speaking speed (default 200, lower = slower)
TTS_VOLUME = 1.0  # volume (0.0 to 1.0)
TTS_VOICE_INDEX = 1  # 0 = default (usually male), 1 = usually female, etc.

# -----------------------------
# Load HuggingFace Model
# -----------------------------
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float16,
    low_cpu_mem_usage=True
).to("cuda")

# -----------------------------
# Assistant Personality & Memory
# -----------------------------
SYSTEM_PROMPT = (
    "You are Mercury, an ancient poet and insightful AI assistant."
    "You answer questions clearly, concisely, and creatively."
)
MAX_MEMORY = 5
conversation_history = []

# -----------------------------
# Queues and threading primitives
# -----------------------------
audio_queue = queue.Queue()
recognized_q = queue.Queue()
keyboard_input_q = queue.Queue()

stop_event = threading.Event()
speaking_event = threading.Event()
tts_complete_event = threading.Event()
tts_complete_event.set()  # initially not speaking

# -----------------------------
# Text-to-Speech logic
# -----------------------------
def speak(text):
    """Run TTS in its own thread, block STT while speaking."""
    def _tts_worker():
        tts_complete_event.clear()  # mark as speaking
        speaking_event.set()  # pause STT
        
        try:
            engine = pyttsx3.init()
            
            # Get available voices
            voices = engine.getProperty('voices')
            
            # Set voice (try to use the configured index, fallback to default)
            if TTS_VOICE_INDEX < len(voices):
                engine.setProperty('voice', voices[TTS_VOICE_INDEX].id)
            
            engine.setProperty("rate", TTS_RATE)
            engine.setProperty("volume", TTS_VOLUME)
            
            # Speak the entire text at once for better reliability
            engine.say(text)
            engine.runAndWait()
            
            # Clean up
            engine.stop()
            del engine
            
        except Exception as e:
            print(f"[TTS error] {e}", file=sys.stderr)
        
        finally:
            speaking_event.clear()  # resume STT
            tts_complete_event.set()  # mark as complete
    
    thread = threading.Thread(target=_tts_worker, daemon=True)
    thread.start()

def list_available_voices():
    """Print all available TTS voices."""
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        print("\n" + "="*60)
        print("Available voices:")
        for i, voice in enumerate(voices):
            print(f"  [{i}] {voice.name} - {voice.id}")
        print("="*60)
        engine.stop()
        del engine
    except Exception as e:
        print(f"Error listing voices: {e}")

# -----------------------------
# Vosk STT setup with silence detection
# -----------------------------
try:
    model_vosk = vosk.Model(VOSK_MODEL_PATH)
except Exception as e:
    print(f"Failed to load Vosk model: {e}", file=sys.stderr)
    raise

def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"[sounddevice status] {status}")
    audio_queue.put(bytes(indata))

def stt_worker():
    """Continuous STT thread with silence detection."""
    rec = vosk.KaldiRecognizer(model_vosk, SAMPLERATE)
    rec.SetWords(False)
    
    print("STT worker started (listening for speech).")
    
    try:
        with sd.RawInputStream(
            samplerate=SAMPLERATE,
            blocksize=BLOCKSIZE,
            dtype='int16',
            channels=1,
            callback=audio_callback
        ):
            partial_text = ""
            last_partial_time = time.time()
            silence_detected = False
            accumulated_text = ""
            
            while not stop_event.is_set():
                # Pause recognition during TTS
                if speaking_event.is_set():
                    time.sleep(0.05)
                    # Reset state when resuming
                    partial_text = ""
                    accumulated_text = ""
                    silence_detected = False
                    last_partial_time = time.time()
                    # Clear audio queue to avoid processing old audio
                    while not audio_queue.empty():
                        try:
                            audio_queue.get_nowait()
                        except queue.Empty:
                            break
                    continue
                
                try:
                    data = audio_queue.get(timeout=0.1)
                except queue.Empty:
                    # Check for silence timeout
                    if (partial_text or accumulated_text) and not silence_detected:
                        elapsed = time.time() - last_partial_time
                        
                        # Check if we have enough content and sufficient silence
                        current_content = (accumulated_text + " " + partial_text).strip()
                        word_count = len(current_content.split())
                        
                        # Filter out if it's just stray short words
                        words = current_content.lower().split()
                        meaningful_words = [w for w in words if w not in IGNORE_WORDS]
                        
                        # Only finalize if we have meaningful content
                        is_substantial = (
                            word_count >= MIN_SPEECH_LENGTH and 
                            len(current_content) >= MIN_CHAR_LENGTH and
                            len(meaningful_words) >= 2  # at least 2 non-filler words
                        )
                        
                        if elapsed > SILENCE_TIMEOUT and is_substantial:
                            # Force finalization after silence
                            final_result = rec.FinalResult()
                            res = json.loads(final_result)
                            text = res.get("text", "")
                            
                            if text or accumulated_text:
                                final_text = (accumulated_text + " " + text).strip()
                                if final_text:
                                    print(f"\n[You said] {final_text}")
                                    recognized_q.put(final_text)
                            
                            # Reset recognizer and state
                            rec = vosk.KaldiRecognizer(model_vosk, SAMPLERATE)
                            rec.SetWords(False)
                            partial_text = ""
                            accumulated_text = ""
                            silence_detected = True
                            last_partial_time = time.time()
                        elif elapsed > SILENCE_TIMEOUT and not is_substantial:
                            # Not substantial enough - just reset without submitting
                            rec = vosk.KaldiRecognizer(model_vosk, SAMPLERATE)
                            rec.SetWords(False)
                            partial_text = ""
                            accumulated_text = ""
                            silence_detected = True
                            last_partial_time = time.time()
                            print("\r" + " " * 80 + "\r", end="", flush=True)  # Clear display
                    
                    continue
                
                if rec.AcceptWaveform(data):
                    res = json.loads(rec.Result())
                    text = res.get("text", "")
                    if text:
                        accumulated_text = (accumulated_text + " " + text).strip()
                        partial_text = ""
                        last_partial_time = time.time()
                        silence_detected = False
                else:
                    pr = json.loads(rec.PartialResult())
                    partial = pr.get("partial", "")
                    if partial and partial != partial_text:
                        partial_text = partial
                        last_partial_time = time.time()
                        silence_detected = False
                        # Show accumulated + current partial
                        display = (accumulated_text + " " + partial_text).strip()
                        print(f"\r[Listening...] {display}                    ", end="", flush=True)
    
    except Exception as e:
        print(f"[STT worker error] {e}", file=sys.stderr)
    
    print("\nSTT worker stopped.")

# -----------------------------
# Keyboard input worker (Windows)
# -----------------------------
def keyboard_worker():
    """Handle keyboard input in background."""
    print("Keyboard worker started.")
    typed_input = ""
    
    while not stop_event.is_set():
        if msvcrt.kbhit():
            char = msvcrt.getch()
            
            try:
                # Handle different key types
                if char == b'\r':  # Enter key
                    if typed_input.strip():
                        print()  # New line after input
                        keyboard_input_q.put(typed_input.strip())
                        typed_input = ""
                elif char == b'\x08':  # Backspace
                    if typed_input:
                        typed_input = typed_input[:-1]
                        # Clear line and reprint
                        print(f"\r>>> {typed_input} ", end="", flush=True)
                elif char == b'\x03':  # Ctrl+C
                    stop_event.set()
                    break
                else:
                    # Regular character
                    try:
                        decoded = char.decode('utf-8')
                        typed_input += decoded
                        print(decoded, end="", flush=True)
                    except:
                        pass
            except Exception as e:
                print(f"\n[Keyboard error] {e}")
        
        time.sleep(0.01)
    
    print("Keyboard worker stopped.")

# -----------------------------
# Agent logic
# -----------------------------
def run_agent(user_input):
    conversation_history.append(f"User: {user_input}")
    recent = conversation_history[-MAX_MEMORY:]
    
    prompt = f"{SYSTEM_PROMPT}\n" + "\n".join(recent) + "\nAssistant:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    output = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id
    )
    
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    
    conversation_history.append(f"Assistant: {reply}")
    return reply

# -----------------------------
# Main loop
# -----------------------------
def main_loop():
    print("\n" + "="*60)
    print("Mercury agent ready!")
    print("- Speak naturally (pause 1.5s to auto-submit)")
    print("- Or type and press Enter")
    print("- Say/type 'exit' to quit")
    print("="*60 + "\n")
    
    # Start background threads
    stt_thread = threading.Thread(target=stt_worker, daemon=True)
    stt_thread.start()
    
    kbd_thread = threading.Thread(target=keyboard_worker, daemon=True)
    kbd_thread.start()
    
    print(">>> ", end="", flush=True)
    
    try:
        while not stop_event.is_set():
            user_input = ""
            
            # Check for voice input
            try:
                user_input = recognized_q.get(timeout=0.1)
            except queue.Empty:
                # Check for keyboard input
                try:
                    user_input = keyboard_input_q.get(timeout=0.1)
                except queue.Empty:
                    continue
            
            if not user_input:
                continue
            
            if user_input.lower() in ("exit", "quit", "stop"):
                print("\nExiting...")
                break
            
            # Generate response
            print("\n[Mercury is thinking...]")
            response = run_agent(user_input)
            print(f"\nMercury: {response}\n")
            
            # Speak response and WAIT for completion
            speak(response)
            tts_complete_event.wait()
            
            print("\n[Ready for next input]")
            print(">>> ", end="", flush=True)
    
    except KeyboardInterrupt:
        print("\n\n(Interrupted) Exiting...")
    
    finally:
        stop_event.set()
        time.sleep(0.3)
        print("\nMercury exited safely.")

if __name__ == "__main__":
    # List available voices on startup
    list_available_voices()
    print("\nTo change voice: edit TTS_VOICE_INDEX at the top of the script")
    print("To change speed: edit TTS_RATE (lower = slower, higher = faster)\n")
    
    main_loop()