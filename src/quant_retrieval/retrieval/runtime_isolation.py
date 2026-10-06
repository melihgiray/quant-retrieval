"""Fail before incompatible native runtimes can be imported on macOS."""

import sys


def ensure_ann_runtime(cached_queries: bool) -> None:
    if sys.platform == "darwin" and (not cached_queries or "torch" in sys.modules):
        raise ValueError("on macOS, export queries first and run --query-cache in a fresh "
                         "process that has not imported torch")


def ensure_encoder_runtime() -> None:
    if sys.platform == "darwin" and "faiss" in sys.modules:
        raise ValueError("encode queries in a fresh process that has not imported FAISS")
