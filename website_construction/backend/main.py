from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from services.family_loader import FamilyRepository


BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
ASSETS_DIR = FRONTEND_DIR / "assets"
FRONTEND_DATA_DIR = FRONTEND_DIR / "data"
PHOTO_DIR = FRONTEND_DATA_DIR / "photos"
DATA_PATH = BASE_DIR / "data" / "family.json"
PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg"}

app = FastAPI(title="Portfolio + Family Tree")

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
