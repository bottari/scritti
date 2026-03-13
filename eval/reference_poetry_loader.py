from pathlib import Path
from typing import List, Optional


def load_human_poems(path: Optional[str] = None) -> List[str]:
    default_path = Path(__file__).resolve().parents[1] / "data" / "human_poetry.txt"
    target = Path(path) if path else default_path
    if not target.exists():
        return []
    if target.is_dir():
        lines: List[str] = []
        for file_path in sorted(target.glob("*.txt")):
            with file_path.open("r", encoding="utf-8") as f:
                lines.extend([line.strip() for line in f if line.strip()])
        return lines
    with target.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

