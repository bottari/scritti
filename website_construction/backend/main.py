from __future__ import annotations
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from llama_cpp import Llama

from models import Person
from services.family_loader import FamilyNotFoundError, FamilyRepository, FamilyValidationError
from services.whitman_pipeline import (
    DEFAULT_FALLBACK_RESPONSE,
    build_formatted_prompt,
    generate_clean_response,
)

# --- PATH CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
ASSETS_DIR = FRONTEND_DIR / "assets"
FRONTEND_DATA_DIR = FRONTEND_DIR / "data"
PHOTO_DIR = FRONTEND_DATA_DIR / "photos"
DATA_PATH = BASE_DIR / "data" / "family.json"
MODEL_PATH = BASE_DIR / "models" / "mini-whitman-q4.gguf"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg"}
VIDEO_EXTENSIONS = {".mp4", ".webm", ".ogg"}
PORTFOLIO_MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
CACHE_CONTROL_NO_CACHE = "no-cache"

class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = CACHE_CONTROL_NO_CACHE
        return response


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
app.mount("/assets", NoCacheStaticFiles(directory=ASSETS_DIR), name="assets")
app.mount("/data", StaticFiles(directory=FRONTEND_DATA_DIR), name="data")


def _page_response(page_name: str) -> FileResponse:
    return FileResponse(
        FRONTEND_DIR / page_name,
        headers={"Cache-Control": CACHE_CONTROL_NO_CACHE},
    )


def _generate_whitman_response(prompt: str, *, retry: bool = False) -> str:
    formatted_prompt = build_formatted_prompt(prompt, retry=retry)

    output = llm(
        formatted_prompt,
        max_tokens=150,
        stop=["<|im_end|>", "<|im_start|>", "<|endoftext|>"],
        temperature=0.6 if retry else 0.7,
        top_p=0.95,
        repeat_penalty=1.2,
        stream=False,
    )

    return output["choices"][0]["text"].strip()


def _get_portfolio_media_type(file_path: Path) -> str:
    if file_path.suffix.lower() in VIDEO_EXTENSIONS:
        return "video"
    return "image"


def _raise_family_http_error(error: Exception) -> None:
    if isinstance(error, FamilyNotFoundError):
        raise HTTPException(status_code=404, detail=str(error)) from error
    if isinstance(error, FamilyValidationError):
        raise HTTPException(status_code=422, detail=str(error)) from error
    raise error


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
    response = generate_clean_response(
        prompt,
        _generate_whitman_response,
        lambda retry_prompt: _generate_whitman_response(retry_prompt, retry=True),
        fallback_response=DEFAULT_FALLBACK_RESPONSE,
    )
    return {"response": response}


# --- FAMILY & PORTFOLIO API ENDPOINTS ---
@app.get("/api/family")
def get_family():
    return [person.model_dump() for person in repository.list_people()]


@app.post("/api/family")
def create_family_person(person: Person):
    try:
        saved_person = repository.create_person(person)
    except Exception as error:  # pragma: no cover - small wrapper over repository behavior
        _raise_family_http_error(error)
    return saved_person.model_dump()


@app.put("/api/family/{person_id}")
def update_family_person(person_id: str, person: Person):
    try:
        saved_person = repository.update_person(person_id, person)
    except Exception as error:  # pragma: no cover - small wrapper over repository behavior
        _raise_family_http_error(error)
    return saved_person.model_dump()


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
            if file_path.is_file() and file_path.suffix.lower() in PORTFOLIO_MEDIA_EXTENSIONS:
                photos.append(
                    {
                        "name": file_path.stem.replace("-", " ").replace("_", " "),
                        "filename": file_path.name,
                        "url": f"/data/photos/{quote(file_path.name)}",
                        "mediaType": _get_portfolio_media_type(file_path),
                    }
                )
    return photos
