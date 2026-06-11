"""Cheap, backend-neutral token estimation (~4 chars/token).

Used only for budgeting and for logging estimate-vs-actual drift; never for
billing. Actual token counts come from the model APIs.
"""

from __future__ import annotations


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)
