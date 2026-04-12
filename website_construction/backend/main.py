from __future__ import annotations

import os
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from llama_cpp import Llama

from services.family_loader import FamilyRepository

# --- PATH CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
ASSETS_DIR = FRONTEND_DIR / "assets"
FRONTEND_DATA_DIR = FRONTEND_DIR / "data"
PHOTO_DIR = FRONTEND_DATA_DIR / "photos"
DATA_PATH = BASE_DIR / "data" / "family.json"
MODEL_PATH = BASE_DIR / "models" / "mini-whitman-q4.gguf"
PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg"}

# --- QWEN 3.5 INITIALIZATION ---
llm = Llama(
    model_path=str(MODEL_PATH),
    n_ctx=512,
    n_threads=2,
    verbose=False,
)

app = FastAPI(title="Portfolio + Family Tree + The Poet")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

repository = FamilyRepository(DATA_PATH)

# --- STATIC FILES ---
app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")
app.mount("/data", StaticFiles(directory=FRONTEND_DATA_DIR), name="data")


def _page_response(page_name: str) -> FileResponse:
    return FileResponse(FRONTEND_DIR / page_name)


THINKING_TAG_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
THINKING_HEADER_RE = re.compile(
    r"^\s*(?:thinking process|thought process|reasoning|analysis|chain of thought|internal reasoning)\s*:",
    re.IGNORECASE,
)
FINAL_MARKER_RE = re.compile(
    r"^\s*(?:final answer|final response|answer|response)\s*:\s*",
    re.IGNORECASE,
)


def _normalize_text(text: str) -> str:
    return (
        text.replace("â€œ", '"')
        .replace("â€", '"')
        .replace("â€˜", "'")
        .replace("â€™", "'")
        .strip()
    )


def _strip_thinking_steps(text: str) -> str:
    cleaned = THINKING_TAG_RE.sub("", text or "")
    cleaned = _normalize_text(cleaned)

    if not cleaned:
        return ""

    lines = cleaned.splitlines()
    for index, line in enumerate(lines):
        if FINAL_MARKER_RE.match(line):
            tail = FINAL_MARKER_RE.sub("", line, count=1).strip()
            remainder = "\n".join(lines[index + 1 :]).strip()
            return "\n".join(part for part in [tail, remainder] if part).strip()

    if THINKING_HEADER_RE.match(cleaned):
        return ""

    return cleaned


def _generate_whitman_response(prompt: str, *, retry: bool = False) -> str:
    system_prompt = (
        "You are the spirit of Walt Whitman. Speak with his poetic soul and transcendentalist wisdom. "
        "Keep your answers short but profound. Never reveal analysis, chain-of-thought, reasoning, steps, "
        "or notes. Output only the final in-character response with no headings or preamble."
    )

    if retry:
        system_prompt += (
            " Do not write 'Thinking Process', numbered lists, bullets, or explanations. "
            "Return exactly one short Whitman-style passage."
        )

    formatted_prompt = (
        f"<|im_start|>system\n"
        f"{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    output = llm(
        formatted_prompt,
        max_tokens=150,
        stop=["<|im_end|>", "<|im_start|>", "</think>", "<|endoftext|>"],
        temperature=0.6 if retry else 0.7,
        top_p=0.95,
        repeat_penalty=1.2,
        stream=False,
    )

    return output["choices"][0]["text"].strip()


# --- PAGE ROUTES ---
@app.get("/")
def root() -> FileResponse:
    return _page_response("index.html")


@app.get("/home")
def home_page() -> FileResponse:
    return _page_response("home.html")


@app.get("/about")
def about_page() -> FileResponse:
    return _page_response("about.html")


@app.get("/portfolio")
def portfolio_page() -> FileResponse:
    return _page_response("portfolio.html")


@app.get("/family")
def family_page() -> FileResponse:
    return _page_response("family.html")


@app.get("/person")
def person_page() -> FileResponse:
    return _page_response("person.html")


@app.get("/poet")
def poet_page() -> FileResponse:
    return _page_response("poet.html")


# --- QWEN 3.5 API ENDPOINT ---
@app.get("/api/whitman")
async def get_whitman_response(prompt: str = Query(..., min_length=1)):
    clean_text = _strip_thinking_steps(_generate_whitman_response(prompt))

    if not clean_text:
        clean_text = _strip_thinking_steps(_generate_whitman_response(prompt, retry=True))

    if not clean_text:
        clean_text = "I sing the bright wire at morning, and the human hand reaching through it for one more living spark."

    return {"response": clean_text}


# --- FAMILY & PORTFOLIO API ENDPOINTS ---
@app.get("/api/family")
def get_family():
    return [person.model_dump() for person in repository.list_people()]


@app.get("/api/person/{person_id}")
def get_person(person_id: str):
    person = repository.get_person(person_id)
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")
    return person.model_dump()


@app.get("/api/tree")
def get_tree():
    return [node.model_dump() for node in repository.get_tree()]


@app.get("/api/portfolio/photos")
def get_portfolio_photos():
    photos = []
    if PHOTO_DIR.exists():
        for file_path in sorted(PHOTO_DIR.iterdir(), key=lambda item: item.name.lower()):
            if file_path.is_file() and file_path.suffix.lower() in PHOTO_EXTENSIONS:
                photos.append(
                    {
                        "name": file_path.stem.replace("-", " ").replace("_", " "),
                        "filename": file_path.name,
                        "url": f"/data/photos/{file_path.name}",
                    }
                )
    return photos
