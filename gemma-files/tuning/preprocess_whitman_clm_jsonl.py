r"""
Build a raw causal-language-modeling JSONL dataset from one poetry .txt file.

The output is intentionally not chat- or instruction-formatted. Each example is
plain source text, optionally prefixed by the tokenizer BOS token, so fine-tuning
teaches the model to continue in the corpus voice by default.

Usage:
    pip install transformers
    python preprocess_whitman_clm_jsonl.py ^
        --input "C:\path\to\leaves_of_grass.txt" ^
        --output "C:\path\to\whitman_clm.jsonl" ^
        --tokenizer "google/gemma-3-4b-pt"
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Iterable

DEFAULT_INPUT = r"C:\Users\micha\Desktop\projects\mercury\Leaves_of_Grass_1882.txt"
DEFAULT_OUTPUT = r"D:\models\whitman_clm_training_data.jsonl"
DEFAULT_TOKENIZER = "google/gemma-3-4b-pt"

MIN_CHUNK_TOKENS = 200
TARGET_CHUNK_TOKENS = 600
MAX_CHUNK_TOKENS = 800
HARD_MAX_TOKENS = 2000
OVERLAP_RATIO = 0.15
REFLECTIVE_PREFIX_RATE = 0.15
RANDOM_SEED = 42
PREVIEW_COUNT = 3


METADATA_PATTERNS = [
    r"^\s*exported from wikisource.*$",
    r"^\s*from wikisource.*$",
    r"^\s*retrieved from .*wikisource.*$",
    r"^\s*this page was last edited.*$",
    r"^\s*downloaded from .*wikisource.*$",
    r"^\s*source:\s*.*$",
    r"^\s*license:\s*.*$",
    r"^\s*category:\s*.*$",
    r"^\s*\[[^\]]*(edit|contents|hide|show)[^\]]*\]\s*$",
    r"^\s*\d{1,2}\s+[A-Z][a-z]+\s+\d{4}\s*$",
    r"^\s*[A-Z][a-z]+\s+\d{1,2},\s+\d{4}\s*$",
    r"^\s*\d{4}-\d{2}-\d{2}\s*$",
    r"^\s*page\s+\d+\s*$",
    r"^\s*\d+\s*$",
]

GUTENBERG_START_RE = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG", re.I)
GUTENBERG_END_RE = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG", re.I)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a raw CLM JSONL dataset from one poetry/prose .txt file."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input .txt corpus file.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output JSONL file.")
    parser.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER,
        help="HuggingFace tokenizer name or local tokenizer path.",
    )
    parser.add_argument("--min-tokens", type=int, default=MIN_CHUNK_TOKENS)
    parser.add_argument("--target-tokens", type=int, default=TARGET_CHUNK_TOKENS)
    parser.add_argument("--max-tokens", type=int, default=MAX_CHUNK_TOKENS)
    parser.add_argument("--hard-max-tokens", type=int, default=HARD_MAX_TOKENS)
    parser.add_argument("--overlap", type=float, default=OVERLAP_RATIO)
    parser.add_argument(
        "--reflective-prefix-rate",
        type=float,
        default=REFLECTIVE_PREFIX_RATE,
        help="Fraction of chunks to prefix with a real short Whitman-like fragment.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--no-bos",
        action="store_true",
        help="Do not prepend tokenizer.bos_token to each JSONL text field.",
    )
    parser.add_argument("--preview-count", type=int, default=PREVIEW_COUNT)
    return parser.parse_args()


def text_tokenizer(tokenizer_or_processor):
    """
    Unsloth may return a multimodal processor for Gemma 4. The text tokenizer
    is nested inside that object, while standalone preprocessing receives a
    normal HuggingFace tokenizer directly.
    """
    if hasattr(tokenizer_or_processor, "encode"):
        return tokenizer_or_processor
    nested = getattr(tokenizer_or_processor, "tokenizer", None)
    if nested is not None and hasattr(nested, "encode"):
        return nested
    raise TypeError(
        f"Could not find a text tokenizer on {type(tokenizer_or_processor).__name__}."
    )


def token_count(tokenizer, text: str) -> int:
    return len(text_tokenizer(tokenizer).encode(text, add_special_tokens=False))


def read_text(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Input .txt file not found: {path}")
    return path.read_text(encoding="utf-8", errors="replace")


def strip_project_headers(text: str) -> str:
    start = GUTENBERG_START_RE.search(text)
    if start:
        line_end = text.find("\n", start.end())
        text = text[line_end + 1 if line_end != -1 else start.end() :]

    end = GUTENBERG_END_RE.search(text)
    if end:
        text = text[: end.start()]

    return text


def looks_like_metadata(line: str) -> bool:
    compact = line.strip()
    if not compact:
        return False

    lowered = compact.lower()
    if any(re.match(pattern, compact, flags=re.I) for pattern in METADATA_PATTERNS):
        return True
    if "wikisource" in lowered or "project gutenberg" in lowered:
        return True
    if lowered.startswith(("proofreaders", "transcriber's note", "transcriber note")):
        return True
    if re.fullmatch(r"[-=_*~]{3,}", compact):
        return True

    return False


def remove_repeated_noise_titles(lines: list[str]) -> list[str]:
    short_line_counts = Counter()
    for line in lines:
        compact = normalize_title_key(line)
        if compact:
            short_line_counts[compact] += 1

    cleaned: list[str] = []
    for line in lines:
        compact = normalize_title_key(line)
        if compact and short_line_counts[compact] > 2:
            continue
        cleaned.append(line)
    return cleaned


def normalize_title_key(line: str) -> str | None:
    compact = re.sub(r"\s+", " ", line.strip())
    if not compact:
        return None
    if len(compact) > 80:
        return None
    if compact.endswith((".", ",", ";", ":", "!", "?")):
        return None
    words = compact.split()
    if len(words) > 8:
        return None
    if re.fullmatch(r"[IVXLCDM]+\.?", compact):
        return compact.upper()
    if compact.isupper() and len(words) >= 2:
        return compact.upper()
    return compact.lower()


def clean_source_text(text: str) -> str:
    """
    Remove source/export scaffolding while preserving the poem's own line breaks,
    punctuation, capitalization, and long sentence structure.
    """
    text = strip_project_headers(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\ufeff", "")

    lines = []
    for line in text.split("\n"):
        line = line.rstrip()
        line = re.sub(r"[ \t]+$", "", line)
        if looks_like_metadata(line):
            continue
        lines.append(line)

    lines = remove_repeated_noise_titles(lines)
    text = "\n".join(lines)

    # Normalize horizontal whitespace, but preserve indentation lightly and keep
    # poetic line breaks intact.
    text = "\n".join(re.sub(r"[ \t]{2,}", " ", line).rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def split_paragraphs(text: str) -> list[str]:
    return [part.strip("\n") for part in re.split(r"\n{2,}", text) if part.strip()]


def split_long_text_on_sentences(text: str) -> list[str]:
    """
    Prefer sentence boundaries while leaving line breaks inside sentences intact.
    Whitman's long flowing sentences are kept unless they exceed the hard limit.
    """
    pieces = re.split(r"(?<=[.!?;:])(\s+)", text)
    sentences: list[str] = []
    current = ""
    for i in range(0, len(pieces), 2):
        piece = pieces[i]
        sep = pieces[i + 1] if i + 1 < len(pieces) else ""
        if not piece:
            current += sep
            continue
        current += piece + sep
        if re.search(r"[.!?;:]\s*$", piece):
            sentences.append(current.rstrip())
            current = ""
    if current.strip():
        sentences.append(current.rstrip())
    return [sentence for sentence in sentences if sentence.strip()]


def split_oversized_unit(tokenizer, text: str, hard_max_tokens: int) -> list[str]:
    if token_count(tokenizer, text) <= hard_max_tokens:
        return [text]

    sentences = split_long_text_on_sentences(text)
    units: list[str] = []
    current: list[str] = []

    for sentence in sentences:
        sentence_tokens = token_count(tokenizer, sentence)
        if sentence_tokens > hard_max_tokens:
            if current:
                units.append("".join(current).strip())
                current = []
            units.extend(split_by_lines_or_tokens(tokenizer, sentence, hard_max_tokens))
            continue

        candidate = "".join(current + [sentence])
        if current and token_count(tokenizer, candidate) > hard_max_tokens:
            units.append("".join(current).strip())
            current = [sentence]
        else:
            current.append(sentence)

    if current:
        units.append("".join(current).strip())

    return [unit for unit in units if unit.strip()]


def split_by_lines_or_tokens(tokenizer, text: str, hard_max_tokens: int) -> list[str]:
    lines = [line for line in text.splitlines(keepends=True) if line.strip()]
    if len(lines) > 1:
        units: list[str] = []
        current = ""
        for line in lines:
            candidate = current + line
            if current and token_count(tokenizer, candidate) > hard_max_tokens:
                units.append(current.strip())
                current = line
            else:
                current = candidate
        if current.strip():
            units.append(current.strip())
        return units

    tokenizer = text_tokenizer(tokenizer)
    ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for start in range(0, len(ids), hard_max_tokens):
        chunk_ids = ids[start : start + hard_max_tokens]
        chunks.append(tokenizer.decode(chunk_ids, skip_special_tokens=True).strip())
    return [chunk for chunk in chunks if chunk]


def build_units(tokenizer, text: str, hard_max_tokens: int) -> list[str]:
    units: list[str] = []
    for paragraph in split_paragraphs(text):
        units.extend(split_oversized_unit(tokenizer, paragraph, hard_max_tokens))
    return units


def choose_overlap_units(
    tokenizer,
    units: list[str],
    chunk_units: list[str],
    end_index: int,
    overlap_tokens: int,
) -> int:
    if overlap_tokens <= 0:
        return end_index

    tokens = 0
    keep_from = end_index
    for idx in range(end_index - 1, end_index - len(chunk_units) - 1, -1):
        if idx < 0:
            break
        unit_tokens = token_count(tokenizer, units[idx])
        if tokens + unit_tokens > overlap_tokens and tokens > 0:
            break
        tokens += unit_tokens
        keep_from = idx
        if tokens >= overlap_tokens:
            break
    return max(keep_from, end_index - len(chunk_units) + 1)


def chunk_units(
    tokenizer,
    units: list[str],
    min_tokens: int,
    target_tokens: int,
    max_tokens: int,
    hard_max_tokens: int,
    overlap_ratio: float,
) -> list[str]:
    chunks: list[str] = []
    i = 0

    while i < len(units):
        current_units: list[str] = []
        current_tokens = 0
        start_i = i

        while i < len(units):
            unit = units[i]
            unit_tokens = token_count(tokenizer, unit)
            separator = "\n\n" if current_units else ""
            candidate_text = separator + unit
            candidate_tokens = token_count(tokenizer, candidate_text)

            would_exceed_max = current_units and current_tokens + candidate_tokens > max_tokens
            reached_target = current_units and current_tokens >= target_tokens
            if would_exceed_max or reached_target:
                break

            current_units.append(unit)
            current_tokens += candidate_tokens if current_units[:-1] else unit_tokens
            i += 1

            if current_tokens >= min_tokens and current_tokens >= target_tokens:
                break

        if not current_units:
            unit = units[i]
            current_units = split_oversized_unit(tokenizer, unit, hard_max_tokens)
            i += 1

        chunk = "\n\n".join(current_units).strip()
        if token_count(tokenizer, chunk) <= hard_max_tokens:
            chunks.append(chunk)
        else:
            chunks.extend(split_oversized_unit(tokenizer, chunk, hard_max_tokens))

        if i >= len(units):
            break

        overlap_tokens = int(max(token_count(tokenizer, chunk) * overlap_ratio, 0))
        next_i = choose_overlap_units(tokenizer, units, current_units, i, overlap_tokens)
        i = max(next_i, start_i + 1)

    return [chunk for chunk in chunks if token_count(tokenizer, chunk) > 0]


def extract_reflective_fragments(text: str) -> list[str]:
    fragments: list[str] = []
    for line in text.splitlines():
        compact = re.sub(r"\s+", " ", line.strip())
        if not compact:
            continue
        if len(compact.split()) < 3:
            continue

        match = re.match(
            r"^((?:And\s+)?(?:I|What|Who|Where|When|Why|How|O)\b[^,;:.!?]{8,90})",
            compact,
        )
        if not match:
            continue

        fragment = match.group(1).strip(" -")
        words = fragment.split()
        if len(words) > 14:
            fragment = " ".join(words[:14])
        fragment = fragment.rstrip(",;:.!?") + ","
        fragments.append(fragment)

    unique = []
    seen = set()
    for fragment in fragments:
        key = fragment.lower()
        if key not in seen:
            unique.append(fragment)
            seen.add(key)
    return unique


def maybe_add_reflective_prefix(
    rng: random.Random,
    chunk: str,
    fragments: list[str],
    rate: float,
) -> str:
    if not fragments or rng.random() >= rate:
        return chunk
    fragment = rng.choice(fragments)
    return f"{fragment}\n{chunk}"


def add_bos(tokenizer, text: str, include_bos: bool) -> str:
    if not include_bos:
        return text
    bos = getattr(text_tokenizer(tokenizer), "bos_token", None)
    if not bos:
        return text
    return bos + text


def write_jsonl(path: Path, samples: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for sample in samples:
            fh.write(json.dumps(sample, ensure_ascii=False) + "\n")


def print_quality_report(tokenizer, samples: list[dict[str, str]], preview_count: int) -> None:
    lengths = [token_count(tokenizer, sample["text"]) for sample in samples]
    if not lengths:
        print("No samples were produced.")
        return

    sorted_lengths = sorted(lengths)
    median = statistics.median(sorted_lengths)

    print("\nDataset quality check")
    print("---------------------")
    print(f"Samples: {len(samples)}")
    print(f"Token length min/median/max: {min(lengths)} / {median:.0f} / {max(lengths)}")

    buckets = [
        ("<200", sum(1 for length in lengths if length < 200)),
        ("200-400", sum(1 for length in lengths if 200 <= length < 400)),
        ("400-800", sum(1 for length in lengths if 400 <= length <= 800)),
        ("801-2000", sum(1 for length in lengths if 800 < length <= 2000)),
        (">2000", sum(1 for length in lengths if length > 2000)),
    ]
    for label, count in buckets:
        print(f"{label:>8}: {count}")

    print(f"\nSample outputs ({min(preview_count, len(samples))})")
    print("----------------")
    if preview_count <= 0:
        return

    sample_indices = evenly_spaced_indices(len(samples), preview_count)
    for rank, idx in enumerate(sample_indices, 1):
        text = samples[idx]["text"]
        print(f"\n[{rank}] index={idx}, tokens={lengths[idx]}")
        print(json.dumps({"text": preview_text(text)}, ensure_ascii=False, indent=2))


def evenly_spaced_indices(total: int, count: int) -> list[int]:
    count = min(count, total)
    if count <= 1:
        return [0]
    return [round(i * (total - 1) / (count - 1)) for i in range(count)]


def preview_text(text: str, max_chars: int = 1200) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n..."


def main() -> None:
    args = parse_args()
    from transformers import AutoTokenizer

    rng = random.Random(args.seed)

    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    print(f"Reading: {input_path}")
    raw_text = read_text(input_path)
    cleaned_text = clean_source_text(raw_text)

    if not cleaned_text:
        raise ValueError("The cleaned corpus is empty. Check the source text and filters.")

    units = build_units(tokenizer, cleaned_text, args.hard_max_tokens)
    chunks = chunk_units(
        tokenizer=tokenizer,
        units=units,
        min_tokens=args.min_tokens,
        target_tokens=args.target_tokens,
        max_tokens=args.max_tokens,
        hard_max_tokens=args.hard_max_tokens,
        overlap_ratio=args.overlap,
    )

    fragments = extract_reflective_fragments(cleaned_text)
    chunks = [
        maybe_add_reflective_prefix(rng, chunk, fragments, args.reflective_prefix_rate)
        for chunk in chunks
    ]

    samples = [
        {"text": add_bos(tokenizer, chunk, include_bos=not args.no_bos)}
        for chunk in chunks
        if token_count(tokenizer, chunk) <= args.hard_max_tokens
    ]

    if not samples:
        raise ValueError("No JSONL samples were produced.")

    write_jsonl(output_path, samples)
    print(f"\nWrote JSONL dataset: {output_path}")
    print_quality_report(tokenizer, samples, args.preview_count)


if __name__ == "__main__":
    main()
