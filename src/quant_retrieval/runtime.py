"""Device selection and seeding, shared by evaluation and training.

Both halves have to agree on these. A run that trains on one device and is
scored after seeding a different way is not the experiment the config claims.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
