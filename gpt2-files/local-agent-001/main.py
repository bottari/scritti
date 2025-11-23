from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pyttsx3
import sounddevice as sd
import vosk
import queue
import json
import re

# -----------------------------
# Model & tokenizer
# -----------------------------
MODEL_PATH = r"models\full_merged_gpt2-finetuned-poetry-mercury-04--copy-attempt"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float16,
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
# Memory
# -----------------------------
MAX_MEMORY = 5
conversation_history = []

# -----------------------------
# Text-to-speech (fixed)
# -----------------------------
def speak(text):
    """
    Speaks the full text reliably, sentence by sentence.
    """
    engine = pyttsx3.init()
    engine.setProperty("rate", 180)
    engine.setProperty("volume", 1.0)

    # Split text into sentences for safe TTS
    chunks = re.split(r'(?<=[.!?]) +', text)
    for chunk in chunks:
        if chunk.strip():
            engine.say(chunk)
            engine.runAndWait()  # process each chunk immediately

    engine.stop()

# -----------------------------
# Offline STT setup using Vosk
# -----------------------------
VOSK_MODEL_PATH = r"C:\Users\micha\Desktop\projects\local-agent-001\vosk-model-en-us-0.22"  # path to model folder
model_vosk = vosk.Model(VOSK_MODEL_PATH)
q = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(bytes(indata))

def listen_vosk():
    """
    Listens offline using Vosk.
    Prints partial results in real-time.
    Returns final recognized text, or empty string if timeout/typing fallback.
    """
    rec = vosk.KaldiRecognizer(model_vosk, 16000)
    print("Listening (offline)... Speak now.")

    final_text = ""
    with sd.RawInputStream(
        samplerate=16000,
        blocksize=4000,
        dtype='int16',
        channels=1,
        callback=audio_callback
    ):
        while True:
            try:
                data = q.get(timeout=2)  # wait for audio, fallback to typing
            except queue.Empty:
                return ""  # fallback to text input

            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get("text", "")
                if text:
                    print(f"\n[You said] {text}")
                    return text
            else:
                partial = json.loads(rec.PartialResult())
                if partial.get("partial"):
                    final_text = partial["partial"]
                    print(f"\r[You said] {final_text}", end="", flush=True)

# -----------------------------
# Agent logic
# -----------------------------
def run_agent(user_input):
    conversation_history.append(f"User: {user_input}")
    short_memory = conversation_history[-MAX_MEMORY:]
    prompt = f"{SYSTEM_PROMPT}\n" + "\n".join(short_memory) + "\nAssistant:"

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

    # Extract only new text (exclude prompt)
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    conversation_history.append(f"Assistant: {reply}")
    return reply

# -----------------------------
# Main loop
# -----------------------------
if __name__ == "__main__":
    print("Mercury agent ready! Speak or type (say 'exit' to quit).\n")

    while True:
        # Try voice input first
        try:
            user_input = listen_vosk()
        except Exception as e:
            print(f"STT error: {e}")
            user_input = ""

        # Fallback to typing if no speech
        if not user_input:
            user_input = input(">>> ")

        if user_input.lower() in ("exit", "quit"):
            break

        response = run_agent(user_input)
        print("\nMercury:", response, "\n")
        speak(response)

    print("Mercury exited safely.")