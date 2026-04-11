from __future__ import annotations

import os
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
# Updated to match your Qwen 3.5 file name
MODEL_PATH = BASE_DIR / "models" / "mini-whitman-q4.gguf"
PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg"}

# --- QWEN 3.5 INITIALIZATION ---
# Using 1 worker and low context to fit in your 1GB Azure RAM
llm = Llama(
    model_path=str(MODEL_PATH),
    n_ctx=512,      
    n_threads=2,    
    verbose=False
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

app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")
app.mount("/data", StaticFiles(directory=FRONTEND_DATA_DIR), name="data")

def _page_response(page_name: str) -> FileResponse:
    return FileResponse(FRONTEND_DIR / page_name)

# --- PAGE ROUTES ---
@app.get("/")
def root() -> FileResponse:
    return _page_response("index.html")

@app.get("/poet")
def poet_page() -> FileResponse:
    return _page_response("poet.html")

# --- QWEN 3.5 API ENDPOINT ---
@app.get("/api/whitman")
async def get_whitman_response(prompt: str = Query(..., min_length=1)):
    """
    Interface for the fine-tuned Qwen 3.5-0.8B model.
    Uses the Qwen ChatML template for high-fidelity persona adherence.
    """
    formatted_prompt = (
        f"<|im_start|>system\n"
        f"You are the spirit of Walt Whitman. Speak with his poetic soul and transcendentalist wisdom. "
        f"Keep your answers short but profound.<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    
    # Qwen 3.5 is very responsive at temperature 0.7
    output = llm(
        formatted_prompt, 
        max_tokens=150, 
        stop=["<|im_end|>", "<|im_start|>"], 
        temperature=0.7
    )
    
    return {"response": output["choices"][0]["text"].strip()}

# --- EXISTING FAMILY TREE ROUTES ---
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