import json
import os
import torch
import warnings
from pathlib import Path
from threading import Thread

from flask import Flask, Response, render_template, request, stream_with_context
from dotenv import load_dotenv
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
    __version__ as TRANSFORMERS_VERSION,
)

warnings.filterwarnings("ignore")
APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent

# Load repo-level .env first, then app-local .env if present.
load_dotenv(REPO_ROOT / ".env")
load_dotenv(APP_DIR / ".env", override=True)


def _env(*keys: str, default: str = "") -> str:
    for key in keys:
        value = os.getenv(key)
        if value is not None:
            return value.strip()
    return default


def _env_bool(*keys: str, default: bool = False) -> bool:
    raw = _env(*keys, default=str(default)).lower()
    return raw in {"1", "true", "yes", "on"}

# CONFIGURATION
HF_TOKEN = _env("HF_TOKEN", "SCRITTI_HF_TOKEN", default="")
BASE_MODEL_NAME = _env("BASE_MODEL_NAME", "SCRITTI_BASE_MODEL_NAME", default="meta-llama/Llama-3.1-8B")
_DEFAULT_ADAPTER_PATH = ""
if "ADAPTER_PATH" in os.environ:
    ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "").strip()
else:
    ADAPTER_PATH = _env("SCRITTI_ADAPTER_PATH", default=_DEFAULT_ADAPTER_PATH)

MAX_NEW_TOKENS = int(_env("MAX_NEW_TOKENS", "SCRITTI_MAX_NEW_TOKENS", default="200"))
TEMPERATURE = float(_env("TEMPERATURE", "SCRITTI_TEMPERATURE", default="0.75"))
TOP_P = float(_env("TOP_P", "SCRITTI_TOP_P", default="0.9"))
TOP_K = int(_env("TOP_K", "SCRITTI_TOP_K", default="25"))
REPETITION_PENALTY = float(_env("REPETITION_PENALTY", "SCRITTI_REPETITION_PENALTY", default="1.2"))

USE_8BIT = _env_bool("USE_8BIT", default=True)
TRUST_REMOTE_CODE = _env_bool("TRUST_REMOTE_CODE", default=True)
HOST = _env("HOST", default="0.0.0.0")
PORT = int(_env("PORT", default="5000"))

app = Flask(__name__, template_folder=str(APP_DIR / "templates"))

model = None
tokenizer = None
stopping_ids = []
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DISPLAY_NAME = BASE_MODEL_NAME


def _hf_kwargs() -> dict:
    return {"token": HF_TOKEN} if HF_TOKEN else {}


def _model_family(name: str) -> str:
    lowered = name.lower()
    if "llama" in lowered:
        return "llama"
    if "qwen" in lowered:
        return "qwen"
    if "gpt2" in lowered:
        return "gpt2"
    return "generic"


def _resolve_local_path(path_str: str) -> Path:
    path_obj = Path(path_str).expanduser()
    if path_obj.is_absolute():
        return path_obj
    return (APP_DIR.parent / path_obj).resolve()


def _adapter_expected_base(adapter_dir: Path) -> str:
    adapter_config = adapter_dir / "adapter_config.json"
    if not adapter_config.exists():
        return ""
    try:
        payload = json.loads(adapter_config.read_text(encoding="utf-8"))
    except Exception:
        return ""
    expected = payload.get("base_model_name_or_path") or payload.get("base_model_name")
    return str(expected).strip() if expected else ""


def _base_names_compatible(requested_base: str, expected_base: str) -> bool:
    def _norm(value: str) -> str:
        return value.replace("\\", "/").strip().lower()

    req = _norm(requested_base)
    exp = _norm(expected_base)
    if not req or not exp:
        return True
    if req == exp:
        return True

    req_leaf = req.split("/")[-1]
    exp_leaf = exp.split("/")[-1]
    return bool(req_leaf and exp_leaf and req_leaf == exp_leaf)


def _build_stopping_ids(local_tokenizer, family: str) -> list[int]:
    # Let non-chat model families (for example GPT-2) use their native generation defaults.
    if family not in {"llama", "qwen"}:
        return []

    ids = []

    eos_id = local_tokenizer.eos_token_id
    if isinstance(eos_id, int):
        ids.append(eos_id)
    elif isinstance(eos_id, list):
        ids.extend([v for v in eos_id if isinstance(v, int)])

    for special_token in ("<|eot_id|>", "<|im_end|>"):
        tok_id = local_tokenizer.convert_tokens_to_ids(special_token)
        if isinstance(tok_id, int) and tok_id >= 0:
            ids.append(tok_id)

    deduped = []
    for value in ids:
        if value not in deduped:
            deduped.append(value)
    return deduped


def _load_base_model() -> AutoModelForCausalLM:
    base_load_kwargs = {
        "trust_remote_code": TRUST_REMOTE_CODE,
        **_hf_kwargs(),
    }

    if torch.cuda.is_available():
        base_load_kwargs["device_map"] = "auto"

    if USE_8BIT and torch.cuda.is_available():
        print(f"Loading base model ({BASE_MODEL_NAME}) with 8-bit quantization...")
        base_load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_has_fp16_weight=False,
        )
    else:
        print(f"Loading base model ({BASE_MODEL_NAME}) without 8-bit quantization...")

    def _raise_helpful_model_error(err: Exception) -> None:
        message = str(err)
        if "does not recognize this architecture" in message or "model_type" in message:
            raise RuntimeError(
                f"Could not load base model: {message}\n"
                f"Installed transformers version: {TRANSFORMERS_VERSION}\n"
                "This model architecture likely needs a newer transformers build.\n"
                "Upgrade in your active environment, then retry:\n"
                "  python -m pip install -U transformers peft accelerate"
            ) from err
        raise RuntimeError(
            f"Could not load base model: {message}\n"
            "Check model name, auth token, and local environment."
        ) from err

    try:
        return AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, **base_load_kwargs)
    except Exception as err:
        if "quantization_config" in base_load_kwargs:
            print("8-bit load failed, retrying without quantization...")
            base_load_kwargs.pop("quantization_config", None)
            try:
                return AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, **base_load_kwargs)
            except Exception as err2:
                _raise_helpful_model_error(err2)
        _raise_helpful_model_error(err)


def load_model():
    global model, tokenizer, stopping_ids, MODEL_DISPLAY_NAME

    print(f"Loading tokenizer from {BASE_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME,
        trust_remote_code=TRUST_REMOTE_CODE,
        **_hf_kwargs(),
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "left"

    family = _model_family(BASE_MODEL_NAME)
    stopping_ids = _build_stopping_ids(tokenizer, family)
    if stopping_ids:
        print(f"Stop token ids: {stopping_ids}")

    base_model = _load_base_model()

    if not ADAPTER_PATH:
        model = base_model
        model.eval()
        MODEL_DISPLAY_NAME = BASE_MODEL_NAME
        print(f"Model ready on {device} (no LoRA adapter).")
        return

    adapter_dir = _resolve_local_path(ADAPTER_PATH)
    if not adapter_dir.exists():
        raise FileNotFoundError(
            f"Adapter path does not exist:\n  {adapter_dir}\nCheck ADAPTER_PATH and try again."
        )
    if not adapter_dir.is_dir():
        raise NotADirectoryError(f"ADAPTER_PATH must point to a directory, got: {adapter_dir}")

    adapter_config = adapter_dir / "adapter_config.json"
    if not adapter_config.exists():
        if (adapter_dir / "config.json").exists():
            raise ValueError(
                f"'{adapter_dir}' looks like a full model checkpoint (has config.json) rather than a LoRA adapter.\n"
                "For full checkpoints, set BASE_MODEL_NAME to this directory and leave ADAPTER_PATH empty.\n"
                "For LoRA adapters, ADAPTER_PATH must contain adapter_config.json and adapter_model.safetensors/bin."
            )
        raise RuntimeError(
            f"Can't find adapter_config.json in:\n  {adapter_dir}\n"
            "This directory is not a valid PEFT LoRA adapter folder."
        )

    expected_base = _adapter_expected_base(adapter_dir)
    if expected_base and not _base_names_compatible(BASE_MODEL_NAME, expected_base):
        raise RuntimeError(
            "Base model and LoRA adapter do not match.\n"
            f"Requested BASE_MODEL_NAME: {BASE_MODEL_NAME}\n"
            f"Adapter expects base model: {expected_base}\n"
            "Choose a matching pair (for example Qwen adapter + Qwen base, Llama adapter + Llama base)."
        )

    print(f"Attaching LoRA adapter from {adapter_dir}...")
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()
    MODEL_DISPLAY_NAME = f"{BASE_MODEL_NAME} + LoRA ({adapter_dir.name})"
    print(f"Model ready on {device}!")


def generate_stream(prompt: str):
    input_text = f"{tokenizer.bos_token}{prompt}" if tokenizer.bos_token else prompt
    inputs = tokenizer(input_text, return_tensors="pt")

    target_device = getattr(model, "device", torch.device(device))
    inputs = inputs.to(target_device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = dict(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        repetition_penalty=REPETITION_PENALTY,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )

    if stopping_ids:
        gen_kwargs["eos_token_id"] = stopping_ids if len(stopping_ids) > 1 else stopping_ids[0]

    generation_error = {"message": None}

    def _run_generate():
        try:
            model.generate(**gen_kwargs)
        except Exception as exc:
            generation_error["message"] = str(exc)
            # Ensure the streamer consumer is unblocked even when generation fails.
            if hasattr(streamer, "end"):
                streamer.end()

    thread = Thread(target=_run_generate, daemon=True)
    thread.start()

    emitted_any = False
    for token_text in streamer:
        emitted_any = True
        yield f"data: {json.dumps({'token': token_text})}\n\n"

    thread.join()

    if generation_error["message"]:
        yield f"data: {json.dumps({'error': generation_error['message']})}\n\n"
    elif not emitted_any:
        yield f"data: {json.dumps({'error': 'Model returned no tokens for this prompt.'})}\n\n"

    yield f"data: {json.dumps({'done': True})}\n\n"


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
    app.run(host=HOST, port=PORT, threaded=True, use_reloader=False, debug=False)
