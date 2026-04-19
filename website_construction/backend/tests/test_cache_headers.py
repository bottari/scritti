from __future__ import annotations

from pathlib import Path


MAIN_FILE = Path(__file__).resolve().parents[1] / "main.py"


def test_pages_set_no_cache_headers() -> None:
    source = MAIN_FILE.read_text(encoding="utf-8")

    assert 'headers={"Cache-Control": CACHE_CONTROL_NO_CACHE}' in source


def test_assets_use_no_cache_static_files() -> None:
    source = MAIN_FILE.read_text(encoding="utf-8")

    assert 'class NoCacheStaticFiles(StaticFiles):' in source
    assert 'response.headers["Cache-Control"] = CACHE_CONTROL_NO_CACHE' in source
    assert 'app.mount("/assets", NoCacheStaticFiles(directory=ASSETS_DIR), name="assets")' in source
