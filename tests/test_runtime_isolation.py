import sys

import pytest

from quant_retrieval.retrieval.runtime_isolation import ensure_ann_runtime, ensure_encoder_runtime


def test_macos_requires_separate_encoder_and_ann_processes(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    with pytest.raises(ValueError, match="query-cache"):
        ensure_ann_runtime(False)
    monkeypatch.setitem(sys.modules, "torch", object())
    with pytest.raises(ValueError, match="fresh process"):
        ensure_ann_runtime(True)
    monkeypatch.delitem(sys.modules, "torch")
    ensure_ann_runtime(True)
    monkeypatch.setitem(sys.modules, "faiss", object())
    with pytest.raises(ValueError, match="fresh process"):
        ensure_encoder_runtime()
    monkeypatch.delitem(sys.modules, "faiss")
    ensure_encoder_runtime()


def test_linux_retains_live_encoding_support(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setitem(sys.modules, "torch", object())
    monkeypatch.setitem(sys.modules, "faiss", object())
    ensure_ann_runtime(False)
    ensure_encoder_runtime()
