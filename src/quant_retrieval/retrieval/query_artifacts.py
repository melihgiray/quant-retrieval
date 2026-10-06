"""Portable validation-query vectors for model-free index benchmarks."""

import re

from quant_retrieval.retrieval.checkpoint import validate_checkpoint_hashes


def validate_query_manifest(manifest: dict) -> None:
    if (not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int
            or manifest["schema_version"] != 1 or manifest.get("kind") != "ann_queries"):
        raise ValueError("unsupported query artifact schema")
    if manifest.get("split") != "val":
        raise ValueError("ANN query artifacts must use the validation split")
    for name in ("queries", "dimensions", "max_length"):
        if type(manifest.get(name)) is not int or manifest[name] <= 0:
            raise ValueError(f"query artifact {name} must be a positive integer")
    if manifest.get("pooling") not in ("mean", "cls") or manifest.get("dtype") != "float32":
        raise ValueError("query artifacts require supported pooling and float32 vectors")
    seed = manifest.get("seed")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("query artifact seed must fit uint32")
    ids = manifest.get("question_ids")
    if (not isinstance(ids, list) or len(ids) != manifest["queries"]
            or any(type(value) is not int or not 0 < value < 2**63 for value in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("query artifact IDs must be unique positive int64 values matching rows")
    for name in ("query_sha256", "vectors_sha256"):
        value = manifest.get(name)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(f"query artifact {name} must be a SHA-256 string")
    validate_checkpoint_hashes(manifest.get("checkpoint_sha256"))
