from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Iterator

THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.IGNORECASE | re.DOTALL)
THINK_OPEN_RE = re.compile(r"<think\b[^>]*>", re.IGNORECASE)
THINK_CLOSE_RE = re.compile(r"</think\s*>", re.IGNORECASE)
THINKING_HEADER_RE = re.compile(
    r"^\s*(?:thinking process|thought process|reasoning|analysis|chain of thought|internal reasoning)\s*:",
    re.IGNORECASE,
)
FINAL_MARKER_RE = re.compile(
    r"^\s*(?:final answer|final response|answer|response)\s*:\s*",
    re.IGNORECASE,
)
REASONING_LIST_RE = re.compile(r"^\s*(?:\d+[.)]|[-*])\s+", re.IGNORECASE)
REASONING_META_RE = re.compile(
    r"^\s*(?:the user|the prompt|i should|i need|i will|we should|we need|let me|first[,:\s]|second[,:\s]|third[,:\s])",
    re.IGNORECASE,
)
REASONING_KEYWORD_RE = re.compile(
    r"\b(?:user|prompt|question|answer|response|reasoning|analysis|step|should|need)\b",
    re.IGNORECASE,
)
LINGERING_REASONING_MARKER_RE = re.compile(
    r"(?:<think\b|</think>|thinking process\s*:|thought process\s*:|chain of thought\s*:|internal reasoning\s*:)",
    re.IGNORECASE,
)

DEFAULT_FALLBACK_RESPONSE = (
    "I sing the bright wire at morning, and the human hand reaching through it for one more living spark."
)


def normalize_text(text: str) -> str:
    return (text or "").strip()


def build_system_prompt(*, retry: bool = False) -> str:
    prompt = (
        "You are the spirit of Walt Whitman. Speak with his poetic soul and transcendentalist wisdom. "
        "Keep your answers short but profound. Never reveal analysis, chain-of-thought, reasoning, steps, "
        "or notes. Output only the final in-character response with no headings or preamble."
    )

    if retry:
        prompt += (
            " Do not write 'Thinking Process', numbered lists, bullets, or explanations. "
            "Return exactly one short Whitman-style passage."
        )

    return prompt


def build_formatted_prompt(prompt: str, *, retry: bool = False) -> str:
    system_prompt = build_system_prompt(retry=retry)
    return (
        f"<|im_start|>system\n"
        f"{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def _looks_like_reasoning_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False

    if THINKING_HEADER_RE.match(stripped) or REASONING_META_RE.match(stripped):
        return True

    return bool(REASONING_LIST_RE.match(stripped) and REASONING_KEYWORD_RE.search(stripped))


def _extract_final_segment(text: str) -> str:
    cleaned = normalize_text(text)
    if not cleaned:
        return ""

    lines = cleaned.splitlines()
    for index, line in enumerate(lines):
        if FINAL_MARKER_RE.match(line):
            tail = FINAL_MARKER_RE.sub("", line, count=1).strip()
            remainder = "\n".join(lines[index + 1 :]).strip()
            return normalize_text("\n".join(part for part in [tail, remainder] if part))

    visible_lines: list[str] = []
    dropping_reasoning = False
    saw_reasoning = False

    for line in lines:
        stripped = line.strip()

        if not stripped:
            if not dropping_reasoning and visible_lines:
                visible_lines.append("")
            continue

        if THINKING_HEADER_RE.match(stripped):
            dropping_reasoning = True
            saw_reasoning = True
            continue

        if dropping_reasoning and _looks_like_reasoning_line(stripped):
            saw_reasoning = True
            continue

        if dropping_reasoning:
            visible_lines.append(stripped)
            dropping_reasoning = False
            continue

        if not saw_reasoning and _looks_like_reasoning_line(stripped):
            dropping_reasoning = True
            saw_reasoning = True
            continue

        visible_lines.append(line.rstrip())

    result = normalize_text("\n".join(visible_lines))
    if LINGERING_REASONING_MARKER_RE.search(result):
        return ""

    return result


def sanitize_model_output(text: str) -> str:
    cleaned = THINK_BLOCK_RE.sub("", text or "")

    unmatched_think = THINK_OPEN_RE.search(cleaned)
    if unmatched_think:
        cleaned = cleaned[: unmatched_think.start()]

    cleaned = THINK_CLOSE_RE.sub("", cleaned)
    return _extract_final_segment(cleaned)


def sanitize_stream_chunks(chunks: Iterable[str]) -> Iterator[str]:
    # Buffer the stream until the visible text is safe so partial reasoning tags
    # can never be flushed to the client between chunk boundaries.
    cleaned = sanitize_model_output("".join(chunk for chunk in chunks if chunk))
    if cleaned:
        yield cleaned


def generate_clean_response(
    prompt: str,
    raw_response_factory: Callable[[str], str],
    raw_retry_response_factory: Callable[[str], str] | None = None,
    *,
    fallback_response: str = DEFAULT_FALLBACK_RESPONSE,
) -> str:
    clean_text = sanitize_model_output(raw_response_factory(prompt))
    if clean_text:
        return clean_text

    if raw_retry_response_factory is not None:
        clean_text = sanitize_model_output(raw_retry_response_factory(prompt))
        if clean_text:
            return clean_text

    return fallback_response
