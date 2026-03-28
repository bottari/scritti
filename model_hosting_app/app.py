import os
import json
import torch
import warnings
from pathlib import Path
from threading import Thread

from flask import Flask, render_template, request, Response, stream_with_context
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer
from peft import PeftModel

warnings.filterwarnings("ignore")

# CONFIGURATION
HF_TOKEN = ""
BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B"
ADAPTER_PATH = r"D:\models\choice-models\llama3-8b-poetry-mercury-26-qlora-8bit-019\final_model"

MAX_NEW_TOKENS = 200
TEMPERATURE = 0.75
TOP_P = 0.9
TOP_K = 25
REPETITION_PENALTY = 1.2

APP_DIR = Path(__file__).resolve().parent
app = Flask(__name__, template_folder=str(APP_DIR / "templates"))

model = None
tokenizer = None
eot_token_id = None
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DISPLAY_NAME = BASE_MODEL_NAME


def load_model():
    global model, tokenizer, eot_token_id, MODEL_DISPLAY_NAME

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU detected. Llama 3.1-8B with QLoRA requires a GPU.")

    print(f"Loading tokenizer from {BASE_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME,
        token=HF_TOKEN or True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    eot_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    print(f"EOT token id: {eot_token_id}")

    print(f"Loading base model ({BASE_MODEL_NAME}) with 8-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_has_fp16_weight=False,
    )

    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            token=HF_TOKEN or True,
        )
    except OSError as e:
        raise RuntimeError(
            f"Could not load base model: {e}\n"
            f"Make sure you accepted the license at https://huggingface.co/{BASE_MODEL_NAME} "
            "and your HF token has access."
        ) from e

    if not os.path.exists(ADAPTER_PATH):
        raise FileNotFoundError(
            f"Adapter not found at:\n  {ADAPTER_PATH}\nCheck the path and try again."
        )

    print(f"Attaching LoRA adapter from {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    MODEL_DISPLAY_NAME = f"{BASE_MODEL_NAME} + LoRA ({Path(ADAPTER_PATH).name})"
    print(f"Model ready on {device}!")


def generate_stream(prompt: str):
    input_text = tokenizer.bos_token + prompt
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    stopping_ids = [tokenizer.eos_token_id, eot_token_id]

    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        repetition_penalty=REPETITION_PENALTY,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=stopping_ids,
        streamer=streamer,
    )

    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    for token_text in streamer:
        yield f"data: {json.dumps({'token': token_text})}\n\n"

    yield f"data: {json.dumps({'done': True})}\n\n"
    thread.join()


@app.route("/")
def index():
    return render_template("index.html", model_name=MODEL_DISPLAY_NAME)


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(force=True)
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return Response("data: {\"error\": \"Empty prompt\"}\n\n", mimetype="text/event-stream")

    return Response(
        stream_with_context(generate_stream(prompt)),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/health")
def health():
    return {
        "status": "ok",
        "device": device,
        "base_model": BASE_MODEL_NAME,
        "adapter_path": ADAPTER_PATH,
        "model": MODEL_DISPLAY_NAME,
    }


if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=5000, threaded=True, use_reloader=False, debug=False)
