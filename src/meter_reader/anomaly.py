"""Consumption check: compare a new reading with the meter's history.

The model only reads the last 3 integer digits (m3), so consumption is computed modulo 1000.
A reading is flagged when consumption is far above the meter's usual level
(robust z-score on past deltas) or when the meter apparently went backwards.
"""
from __future__ import annotations

import statistics


def final_value(rec: dict) -> str | None:
    return rec.get("corrected_value") or rec.get("value")


def check(new_value: str, history: list[dict]) -> tuple[int | None, str | None]:
    """history: previous readings of the same meter, newest first."""
    prev = [final_value(r) for r in history if final_value(r)]
    if not prev:
        return None, None
    delta = (int(new_value) - int(prev[0])) % 1000
    if delta > 500:
        return delta, "meter_went_backwards_or_misread"
    past = [(int(a) - int(b)) % 1000 for a, b in zip(prev, prev[1:])]
    past = [d for d in past if d <= 500]
    if len(past) >= 3:
        med = statistics.median(past)
        mad = statistics.median([abs(d - med) for d in past]) or 1
        if (delta - med) / (1.4826 * mad) > 3.5:
            return delta, "unusually_high_consumption"
    return delta, None
