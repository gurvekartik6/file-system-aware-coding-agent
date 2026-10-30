"""Shared Gemini model configuration."""

import os
from typing import Any


def sampling_kwargs() -> dict[str, Any]:
    """Return optional sampling settings without forcing fixed-default models."""
    value = os.getenv("GEMINI_TEMPERATURE")

    if value is None or not value.strip():
        return {}

    try:
        temperature = float(value)
    except ValueError as exc:
        raise RuntimeError(
            "GEMINI_TEMPERATURE must be a number between 0 and 2."
        ) from exc

    if not 0 <= temperature <= 2:
        raise RuntimeError(
            "GEMINI_TEMPERATURE must be a number between 0 and 2."
        )

    return {"temperature": temperature}
