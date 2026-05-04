"""
Completion-style poetry generator for a merged fp16 Gemma 4 poetry fine-tune.

This script is intentionally not a chat loop. It loads the merged model and
continues a title, seed line, or prompt as raw poetry text.

Usage
-----
    python gemma-files\\generation\\gemma-chat-merged-fp16.py
    python gemma-files\\generation\\gemma-chat-merged-fp16.py --title "Song of the Open Road"
    python gemma-files\\generation\\gemma-chat-merged-fp16.py --prompt "I hear the train at dusk"
    python gemma-files\\generation\\gemma-chat-merged-fp16.py --model D:\\models\\gemma4-poetry-finetune-whitmanv6\\merged_fp16
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import re
import sys
import warnings
from threading import Thread

# Keep Unsloth/Torch from compiling kernels during interactive inference.
# This is a runtime stability flag, not a model or generation setting.
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")

import torch

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("unsloth").setLevel(logging.ERROR)

from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from unsloth import FastLanguageModel


DEFAULT_MODEL = r"D:\models\gemma4-poetry-finetune-whitmanv6\merged_fp16"
MAX_SEQ_LENGTH = 2048

# Generation defaults. These affect inference only, not training.
MAX_NEW_TOKENS = 360
TEMPERATURE = 0.85
TOP_P = 0.95
TOP_K = 64
REP_PENALTY = 1.12
NO_REPEAT_NGRAM_SIZE = 4

STOP_STRINGS = [
    "<start_of_turn>",
    "<start-of-turn>",
    "<end_of_turn>",
    "<end-of-turn>",
    "<|channel>",
    "<channel|>",
    "<start-new-thread>",
    "<blockquote",
    "</blockquote",
    "<br",
    "\nUser:",
    "\nAssistant:",
]


def _supports_colour() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


USE_COLOUR = _supports_colour()


class C:
    RESET = "\033[0m" if USE_COLOUR else ""
    BOLD = "\033[1m" if USE_COLOUR else ""
    CYAN = "\033[36m" if USE_COLOUR else ""
    GREEN = "\033[32m" if USE_COLOUR else ""
    GREY = "\033[90m" if USE_COLOUR else ""
    MAGENTA = "\033[35m" if USE_COLOUR else ""
    RED = "\033[31m" if USE_COLOUR else ""
    WHITE = "\033[97m" if USE_COLOUR else ""
    YELLOW = "\033[33m" if USE_COLOUR else ""


TERM_WIDTH = 72


def print_rule(char: str = "-", colour: str = C.GREY) -> None:
    print(f"{colour}{char * TERM_WIDTH}{C.RESET}")


def print_banner(model_path: str, dtype_name: str) -> None:
    print_rule("=", C.CYAN)
    print(f"{C.CYAN}{C.BOLD}  Gemma 4 poetry fine-tune | completion mode{C.RESET}")
    print(f"{C.GREY}  model : {model_path}{C.RESET}")
    print(f"{C.GREY}  dtype : {dtype_name} | load_in_4bit=False{C.RESET}")
    print(f"{C.GREY}  input : poem title, seed line, or prompt | /exit to quit{C.RESET}")
    print_rule("=", C.CYAN)
    print()


SPECIAL_TOKEN_NOISE = re.compile(
    r"<\|[^>]+\|?>|</?[a-z][a-z0-9_-]*(?:\s+[^>]*)?/?>|"
    r"\[/?(?:channel|think|end_of_turn|bos|eos)[^\]]*\]",
    re.IGNORECASE,
)


def clean_generated_text(text: str) -> str:
    text = SPECIAL_TOKEN_NOISE.sub("", text)
    text = re.sub(r"^\s*(?:Assistant|Model|User|Poet):\s*", "", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text


def first_stop_index(text: str) -> int:
    matches = [text.find(stop) for stop in STOP_STRINGS if stop in text]
    return min(matches) if matches else -1


def dtype_from_name(name: str):
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return None


def build_seed_text(title: str = "", prompt: str = "") -> str:
    title = title.strip()
    prompt = prompt.strip()

    if title and prompt:
        return f"{title}\n\n{prompt}"
    if title:
        return f"{title}\n\n"
    return prompt


def encode_seed(tokenizer, seed_text: str, device: torch.device) -> torch.Tensor:
    if not seed_text.strip():
        raise ValueError("Seed text is empty.")

    bos = getattr(tokenizer, "bos_token", None) or ""
    text = seed_text
    if bos and not text.startswith(bos):
        text = bos + text

    encoded = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    )
    input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    return input_ids.to(device)


class StopOnStrings(StoppingCriteria):
    def __init__(self, tokenizer, prompt_length: int, stop_strings: list[str]) -> None:
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.stop_strings = stop_strings

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        generated = input_ids[0, self.prompt_length :]
        if generated.numel() == 0:
            return False
        tail = generated[-80:]
        text = self.tokenizer.decode(tail, skip_special_tokens=False)
        return any(stop in text for stop in self.stop_strings)


def stop_token_ids(tokenizer) -> list[int]:
    ids = []
    for token_id in [getattr(tokenizer, "eos_token_id", None)]:
        if isinstance(token_id, int) and token_id >= 0:
            ids.append(token_id)
    return sorted(set(ids))


def stream_completion(streamer) -> str:
    pending = ""
    output = ""
    keep_chars = max(len(stop) for stop in STOP_STRINGS)

    for chunk in streamer:
        pending += chunk
        stop_at = first_stop_index(pending)
        if stop_at != -1:
            emit = clean_generated_text(pending[:stop_at])
            if emit:
                print(f"{C.WHITE}{emit}{C.RESET}", end="", flush=True)
                output += emit
            pending = ""
            break

        if len(pending) > keep_chars:
            emit_raw = pending[:-keep_chars]
            pending = pending[-keep_chars:]
            emit = clean_generated_text(emit_raw)
            if emit:
                print(f"{C.WHITE}{emit}{C.RESET}", end="", flush=True)
                output += emit

    if pending:
        stop_at = first_stop_index(pending)
        if stop_at != -1:
            pending = pending[:stop_at]
        emit = clean_generated_text(pending)
        if emit:
            print(f"{C.WHITE}{emit}{C.RESET}", end="", flush=True)
            output += emit

    return output.strip()


def generate_completion(
    model,
    tokenizer,
    seed_text: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> str:
    input_ids = encode_seed(tokenizer, seed_text, model.device)
    attention_mask = torch.ones_like(input_ids, device=model.device)
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=False,
    )
    eos_token_ids = stop_token_ids(tokenizer)

    generation_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=eos_token_ids or tokenizer.eos_token_id,
        stopping_criteria=StoppingCriteriaList(
            [StopOnStrings(tokenizer, input_ids.shape[-1], STOP_STRINGS)]
        ),
    )

    thread = Thread(target=lambda: model.generate(**generation_kwargs))
    thread.start()
    output = stream_completion(streamer)
    thread.join()
    print(f"{C.RESET}")
    return output


def prompt_for_seed() -> str:
    print(f"{C.CYAN}{C.BOLD}Seed >{C.RESET} ", end="", flush=True)
    return input().strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continue a poem with a merged fp16 Gemma 4 poetry fine-tune."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Path to the merged_fp16 model folder.",
    )
    parser.add_argument(
        "--dtype",
        choices=["fp16", "bf16", "auto"],
        default="fp16",
        help="Torch dtype to use when loading the merged model.",
    )
    parser.add_argument("--title", default="", help="Optional poem title seed.")
    parser.add_argument("--prompt", default="", help="Optional poem body seed.")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=TOP_P)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--repetition-penalty", type=float, default=REP_PENALTY)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=NO_REPEAT_NGRAM_SIZE)
    args = parser.parse_args()

    model_path = args.model
    if not Path(model_path).exists():
        print(f"{C.RED}Model path does not exist: {model_path}{C.RESET}")
        print(f"{C.GREY}Pass --model D:\\path\\to\\merged_fp16 if yours is elsewhere.{C.RESET}")
        raise SystemExit(1)

    load_dtype = dtype_from_name(args.dtype)

    print(f"\n{C.GREY}Loading merged model from {model_path} ...{C.RESET}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,
        dtype=load_dtype,
    )
    FastLanguageModel.for_inference(model)
    print(f"{C.GREEN}  - Model ready{C.RESET}\n")

    print_banner(model_path, args.dtype)

    initial_seed = build_seed_text(args.title, args.prompt)
    if initial_seed.strip():
        print(f"{C.MAGENTA}{C.BOLD}Seed{C.RESET}\n{initial_seed}\n")
        print(f"{C.MAGENTA}{C.BOLD}Continuation{C.RESET}\n", end="", flush=True)
        generate_completion(
            model=model,
            tokenizer=tokenizer,
            seed_text=initial_seed,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )
        print()
        return

    while True:
        try:
            seed_text = prompt_for_seed()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{C.GREY}Goodbye.{C.RESET}\n")
            break

        if not seed_text:
            continue
        if seed_text.lower() in ("/exit", "/quit", "exit", "quit"):
            print(f"{C.GREY}Goodbye.{C.RESET}\n")
            break

        print(f"\n{C.MAGENTA}{C.BOLD}Continuation{C.RESET}\n", end="", flush=True)
        generate_completion(
            model=model,
            tokenizer=tokenizer,
            seed_text=seed_text,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )
        print("\n")
        print_rule()
        print()


if __name__ == "__main__":
    main()
