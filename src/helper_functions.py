from typing import Sequence


def fail_if_no_successes(
    label: str,
    total: int,
    successes: int,
    errors: Sequence,
) -> None:
    """Raise RuntimeError if a per-item loop attempted N items and zero succeeded."""
    if total > 0 and successes == 0:
        sample = list(errors)[:3]
        raise RuntimeError(
            f"{label}: 0/{total} succeeded. First errors: {sample}"
        )
