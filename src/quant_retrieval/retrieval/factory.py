"""Build a retriever, or a whole pipeline, from a config dictionary.

Hybrid and reranking retrievers wrap others, so a spec is a tree and this
recurses. Keeping the shape identical at every level means a pipeline is
described entirely by its config file, and the committed result carries that
whole tree as provenance rather than just the outermost name.

It lives in the package rather than in a script because more than one entry
point needs it: evaluation builds pipelines, and so does the profiler.
"""

from __future__ import annotations

from typing import Any

from quant_retrieval.retrieval.bm25 import BM25Retriever
from quant_retrieval.retrieval.dense import DenseRetriever
from quant_retrieval.retrieval.hybrid import HybridRetriever
from quant_retrieval.retrieval.rerank import RerankingRetriever


def build_retriever(config: dict[str, Any]):
    """Build a retriever from a spec, recursing into nested ones.

    Hybrid and reranking retrievers wrap others, so a spec is a tree. Keeping the
    same shape at every level means a pipeline is described entirely by its
    config file, and the committed result carries that whole tree as provenance.
    """
    name = config["retriever"]
    parameters = dict(config.get("parameters", {}))
    if name == "bm25":
        return BM25Retriever(**parameters)
    if name == "dense":
        return DenseRetriever(**parameters)
    if name == "hybrid":
        nested = [build_retriever(spec) for spec in parameters.pop("retrievers")]
        return HybridRetriever(nested, **parameters)
    if name == "rerank":
        base = build_retriever(parameters.pop("base"))
        return RerankingRetriever(base, **parameters)
    raise ValueError(f"unknown retriever: {name}")
