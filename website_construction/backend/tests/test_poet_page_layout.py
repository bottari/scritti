from __future__ import annotations

from pathlib import Path


POET_PAGE = Path(__file__).resolve().parents[2] / "frontend" / "poet.html"


def test_poet_page_uses_mobile_safe_viewport_height() -> None:
    page = POET_PAGE.read_text(encoding="utf-8")

    assert "min-height: 100vh;" in page
    assert "height: 100vh;" in page
    assert "@supports (height: 100dvh)" in page
    assert "min-height: 100dvh;" in page
    assert "height: 100dvh;" in page


def test_poet_page_adds_bottom_safe_area_padding_to_composer() -> None:
    page = POET_PAGE.read_text(encoding="utf-8")

    assert "padding-bottom: calc(1.25rem + env(safe-area-inset-bottom));" in page
