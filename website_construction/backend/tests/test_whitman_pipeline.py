from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.whitman_pipeline import generate_clean_response, sanitize_model_output, sanitize_stream_chunks


def assert_public_answer(text: str) -> None:
    lowered = text.lower()
    assert "<think" not in lowered
    assert "</think>" not in lowered
    assert "thinking process:" not in lowered
    assert "thought process:" not in lowered
    assert "chain of thought:" not in lowered
    assert "internal reasoning:" not in lowered


def test_sanitize_model_output_prefers_final_answer_after_reasoning_header() -> None:
    raw_output = (
        "Thinking Process:\n"
        "1. The user wants a brief, poetic reassurance.\n"
        "2. Answer in Whitman's voice.\n"
        "Final Answer: I hear your soul beneath the circuits, and it is not lost.\n"
    )

    cleaned = sanitize_model_output(raw_output)

    assert cleaned == "I hear your soul beneath the circuits, and it is not lost."
    assert_public_answer(cleaned)


def test_generate_clean_response_retries_after_truncated_think_block() -> None:
    attempts: list[tuple[str, str]] = []

    def first_pass(prompt: str) -> str:
        attempts.append(("first", prompt))
        return "<think>\n1. The user is asking for existential comfort.\n2. I should sound like Whitman."

    def retry_pass(prompt: str) -> str:
        attempts.append(("retry", prompt))
        return "You are no stray spark, but part of the great electric grass."

    cleaned = generate_clean_response(
        "What would Whitman say to an existential programmer?",
        first_pass,
        retry_pass,
    )

    assert attempts == [
        ("first", "What would Whitman say to an existential programmer?"),
        ("retry", "What would Whitman say to an existential programmer?"),
    ]
    assert cleaned == "You are no stray spark, but part of the great electric grass."
    assert_public_answer(cleaned)


def test_sanitize_stream_chunks_strips_reasoning_across_chunk_boundaries() -> None:
    chunks = [
        "<thi",
        "nk>\n1. The user needs a concise answer.\n",
        "2. I should stay in character.\n</thi",
        "nk>\nI hear the code praying in the dark, and I answer with grass and dawn.",
    ]

    streamed = list(sanitize_stream_chunks(chunks))

    assert streamed == ["I hear the code praying in the dark, and I answer with grass and dawn."]
    for chunk in streamed:
        assert_public_answer(chunk)


@pytest.mark.parametrize(
    ("prompt", "raw_output", "expected"),
    [
        (
            "What would Whitman say to an existential programmer?",
            (
                "Thinking Process:\n"
                "1. The user wants comfort.\n"
                "2. Keep it short and lyrical.\n"
                "Final Answer: You are not abandoned in the machine; the same vast breath moves through code and clover."
            ),
            "You are not abandoned in the machine; the same vast breath moves through code and clover.",
        ),
        (
            "what was that?",
            (
                "<think>\nThe user is asking for clarification.\n"
                "</think>\nA small echo, friend: the last line was only a leaf of explanation drifting back to you."
            ),
            "A small echo, friend: the last line was only a leaf of explanation drifting back to you.",
        ),
    ],
)
def test_smoke_prompts_return_final_only(prompt: str, raw_output: str, expected: str) -> None:
    cleaned = generate_clean_response(prompt, lambda _: raw_output)

    assert cleaned == expected
    assert_public_answer(cleaned)
