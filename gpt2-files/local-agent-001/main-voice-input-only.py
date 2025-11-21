from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import sounddevice as sd
import vosk
import queue
import json
import time
from collections import deque

# -----------------------------
# Model & tokenizer with error handling
# -----------------------------
MODEL_PATH = r"models\full_merged_gpt2-finetuned-poetry-mercury-04--copy-attempt"

def load_model():
    """Load model with proper error handling"""
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, using CPU (will be slow)")
        device = "cpu"
        dtype = torch.float32
    else:
        device = "cuda"
        dtype = torch.float16
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    ).to(device)
    
    model.eval()  # Set to evaluation mode
    return model, tokenizer, device

try:
    model, tokenizer, device = load_model()
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# -----------------------------
# System instructions / personality
# -----------------------------
SYSTEM_PROMPT = (
    "You are Mercury, an ancient poet and insightful AI assistant. "
)

# -----------------------------
# Conversation memory with bounded size
# -----------------------------
MAX_MEMORY = 5
conversation_history = deque(maxlen=MAX_MEMORY * 2)  # *2 for user+assistant pairs

# -----------------------------
# Stopword cleanup for ghost words
# -----------------------------
STOPWORDS = {"the", "a", "an", "um", "uh", "er"}

def clean_recognized_text(text):
    """Remove filler words and clean up recognized text"""
    if not text:
        return ""
    
    words = text.split()
    
    # Remove leading stopwords/short words
    while words and (len(words[0]) <= 2 or words[0].lower() in STOPWORDS):
        words.pop(0)
    
    # Remove trailing stopwords/short words
    while words and (len(words[-1]) <= 2 or words[-1].lower() in STOPWORDS):
        words.pop()
    
    return " ".join(words).strip()

# -----------------------------
# Offline STT setup using Vosk
# -----------------------------
VOSK_MODEL_PATH = r"C:\Users\micha\Desktop\projects\local-agent-001\vosk-model-en-us-0.22"

try:
    model_vosk = vosk.Model(VOSK_MODEL_PATH)
except Exception as e:
    print(f"Error loading Vosk model: {e}")
    print("Falling back to text-only mode")
    model_vosk = None

q = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(f"Audio status: {status}", flush=True)
    q.put(bytes(indata))

def listen_vosk(timeout=5):
    """Listen for speech with shorter timeout"""
    if model_vosk is None:
        return None
    
    rec = vosk.KaldiRecognizer(model_vosk, 16000)
    
    last_speech_time = time.time()
    silence_threshold = 1.5  # seconds of silence before considering input complete
    
    try:
        with sd.RawInputStream(samplerate=16000, blocksize=2048, dtype='int16',
                               channels=1, callback=audio_callback):
            while True:
                # Check for timeout
                if time.time() - last_speech_time > timeout:
                    return None
                
                try:
                    data = q.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    text = result.get("text", "")
                    if text:
                        text = clean_recognized_text(text)
                        if text:  # Only return if we have actual content
                            # Clear the queue
                            while not q.empty():
                                try:
                                    q.get_nowait()
                                except queue.Empty:
                                    break
                            print(f"\n[You said] {text}")
                            return text
                        last_speech_time = time.time()
                else:
                    # Show partial results
                    partial = json.loads(rec.PartialResult())
                    partial_text = partial.get("partial", "")
                    if partial_text:
                        print(f"[Listening...] {partial_text}", end="\r", flush=True)
                        last_speech_time = time.time()
                    elif time.time() - last_speech_time > silence_threshold and partial_text == "":
                        # Silence detected after previous speech
                        continue
    
    except Exception as e:
        print(f"\nAudio error: {e}")
        return None

# -----------------------------
# Agent function with improved parsing
# -----------------------------
def run_agent(user_input):
    """Generate response with improved prompt handling"""
    conversation_history.append(f"User: {user_input}")
    
    # Build prompt from recent history
    memory = "\n".join(conversation_history)
    prompt = f"{SYSTEM_PROMPT}\n\n{memory}\nAssistant:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():  # Disable gradient computation for inference
        output = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=True
        )
    
    # Decode only the new tokens
    generated_text = tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], 
                                      skip_special_tokens=True)
    
    # Clean up the response
    assistant_reply = generated_text.strip()
    
    # Stop at the first occurrence of "User:" to prevent prompt leakage
    if "User:" in assistant_reply:
        assistant_reply = assistant_reply.split("User:")[0].strip()
    
    # Remove any remaining "Assistant:" prefix
    if assistant_reply.startswith("Assistant:"):
        assistant_reply = assistant_reply[len("Assistant:"):].strip()
    
    conversation_history.append(f"Assistant: {assistant_reply}")
    return assistant_reply

# -----------------------------
# Main loop with improved UX
# -----------------------------
if __name__ == "__main__":
    print("\n" + "="*50)
    print("Mercury Agent Ready!")
    print("="*50)
    print("Commands:")
    print("  - Type your message and press Enter")
    print("  - Type '/voice' to switch to voice input")
    print("  - Type '/exit' to quit")
    print("  - Type '/clear' to reset conversation")
    print("="*50 + "\n")
    
    voice_mode = False
    
    try:
        while True:
            user_input = None
            
            # Voice mode - try voice input with short timeout
            if voice_mode and model_vosk is not None:
                print("🎤 [Voice mode - speak now, or just press Enter to type]")
                try:
                    user_input = listen_vosk(timeout=5)
                except KeyboardInterrupt:
                    user_input = None
                    voice_mode = False
                    print("\n[Switched to text mode]")
                except Exception as e:
                    print(f"Voice input error: {e}")
                    voice_mode = False
            
            # Fall back to text input if no voice input received
            if not user_input:
                try:
                    user_input = input(">>> " if not voice_mode else ">>> ").strip()
                except KeyboardInterrupt:
                    print("\n")
                    break
            
            if not user_input:
                continue
            
            # Handle commands (require / prefix to avoid accidental exits)
            if user_input.lower() in ["/exit", "/quit", "/bye"]:
                print("Goodbye!")
                break
            
            if user_input.lower() == "/clear":
                conversation_history.clear()
                print("Conversation history cleared.\n")
                continue
            
            if user_input.lower() == "/voice":
                if model_vosk is None:
                    print("Voice input not available (Vosk model not loaded)\n")
                else:
                    voice_mode = not voice_mode
                    status = "enabled" if voice_mode else "disabled"
                    print(f"Voice mode {status}\n")
                continue
            
            # Generate response
            try:
                response = run_agent(user_input)
                print(f"\n💫 Mercury: {response}\n")
            except Exception as e:
                print(f"Error generating response: {e}\n")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        print("Mercury exited safely.")