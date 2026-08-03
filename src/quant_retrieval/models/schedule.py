"""Learning rate schedule for fine-tuning.

Warm up linearly, then decay linearly to zero. The warmup matters more than it
looks: at step zero the encoder produces sensible embeddings already, and a full
learning rate straight into a contrastive loss with random in-batch negatives
will wreck them before the gradient signal means anything.
"""

from __future__ import annotations


def linear_warmup_decay(step: int, total_steps: int, warmup_steps: int) -> float:
    """Multiplier on the base learning rate for a given step."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if warmup_steps < 0 or warmup_steps > total_steps:
        raise ValueError("warmup_steps must be between 0 and total_steps")

    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    remaining = total_steps - step
    return max(0.0, remaining / max(1, total_steps - warmup_steps))
