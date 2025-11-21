from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pyttsx3
import sounddevice as sd
import vosk
import queue
import json
import threading
import re

# -----------------------------
# Model & tokenizer
# -----------------------------
MODEL_PATH = r"models\full_merged_gpt2-finetuned-poetry-mercury-04--copy-attempt"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float16,  # GPU FP16
    low_cpu_mem_usage=True
).to("cuda")

# -----------------------------
# System instructions / personality
# -----------------------------
SYSTEM_PROMPT = (
    "You are Mercury, a poetic and insightful AI assistant. "
    "You answer questions clearly, concisely, and creatively."
)

# -----------------------------
# Conversation memory
# -----------------------------
MAX_MEMORY = 5
conversation_history = []

# -----------------------------
# Text-to-speech setup (interruptible)
# -----------------------------
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 180)
tts_engine.setProperty('volume', 1.0)

speech_queue = queue.Queue()
stop_signal = threading.Event()
shutdown_signal = threading.Event()

def tts_worker():
    while not shutdown_signal.is_set():
        try:
            text = speech_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if text is None:
            break

        stop_signal.clear()
        chunks = re.split(r'(?<=[.!?]) +', text)
        for chunk in chunks:
            if stop_signal.is_set() or shutdown_signal.is_set():
                break
            tts_engine.say(chunk)
            tts_engine.runAndWait()

tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

def speak(text):
    stop_speech()
    speech_queue.put(text)

def stop_speech():
    stop_signal.set()
    with speech_queue.mutex:
        speech_queue.queue.clear()

# -----------------------------
# Offline STT setup using Vosk
# -----------------------------
VOSK_MODEL_PATH = r"C:\Users\micha\Desktop\projects\local-agent-001\vosk-model-en-us-0.22"
model_vosk = vosk.Model(VOSK_MODEL_PATH)
q = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(status, flush=True)
    q.put(bytes(indata))

def listen_vosk():
    rec = vosk.KaldiRecognizer(model_vosk, 16000)
    print("Listening (offline)... Speak now.")

    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                           channels=1, callback=audio_callback):
        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get("text", "")
                if text:
                    print(f"\n[You said] {text}")
                    return text
            else:
                partial = json.loads(rec.PartialResult())
                if partial.get("partial"):
                    print(f"[Partial] {partial['partial']}", end="\r", flush=True)

# -----------------------------
# Agent function
# -----------------------------
def run_agent(user_input):
    conversation_history.append(f"User: {user_input}")

    memory = "\n".join(conversation_history[-MAX_MEMORY:])
    prompt = f"{SYSTEM_PROMPT}\n{memory}\nAssistant:"

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

    text = tokenizer.decode(output[0], skip_special_tokens=True)

    if "Assistant:" in text:
        assistant_reply = text.split("Assistant:")[-1].strip()
    else:
        assistant_reply = text.strip()

    conversation_history.append(f"Assistant: {assistant_reply}")
    return assistant_reply

# -----------------------------
# Main loop
# -----------------------------
if __name__ == "__main__":
    print("Mercury agent ready! Speak or type (say 'exit' to quit).\n")

    try:
        while True:
            try:
                user_input = listen_vosk()
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"STT error: {e}")
                user_input = input(">>> ")

            if not user_input:
                user_input = input(">>> ")

            if user_input.lower() in ["exit", "quit"]:
                break

            stop_speech()
            response = run_agent(user_input)
            print("\nMercury:", response, "\n")
            speak(response)

    finally:
        stop_speech()
        shutdown_signal.set()
        speech_queue.put(None)  # unblock TTS thread
        tts_thread.join()
        print("Mercury exited safely.")