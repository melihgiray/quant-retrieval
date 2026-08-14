"""Embed the corpus once and save it.

    python scripts/export_index.py --checkpoint checkpoints/minilm_tuned/epoch-3

Everything downstream needs these vectors and none of it needs them recomputed.
The ANN study sweeps a dozen index settings over the same embeddings, and the
hosted demo cannot afford to embed 26,152 documents at startup on two CPU cores.
Both read what this writes.

Saves float32 and float16 side by side. float16 halves the file at a cost that
should be nil, since the vectors are normalized and live in [-1, 1] where half
precision has plenty of resolution, but "should be nil" is not a measurement, so
both are written and the evaluation decides.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from quant_retrieval.retrieval.dense import DenseRetriever  # noqa: E402
from quant_retrieval.runtime import set_seed  # noqa: E402


def current_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True
    )
    return completed.stdout.strip() or "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/minilm_tuned/epoch-3"))
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="a corpus parquet other than the default, for the scaling study",
    )
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    set_seed(args.seed)
    corpus = pd.read_parquet(args.corpus or args.data / "corpus.parquet")
    retriever = DenseRetriever(
        str(args.checkpoint), batch_size=args.batch_size, max_length=args.max_length
    )

    started = time.perf_counter()
    embeddings = retriever._encode(corpus["text"].tolist())
    seconds = time.perf_counter() - started

    args.out.mkdir(parents=True, exist_ok=True)
    answer_ids = corpus["answer_id"].astype(np.int64).to_numpy()
    np.save(args.out / "answer_ids.npy", answer_ids)

    sizes = {}
    for name, dtype in (("fp32", np.float32), ("fp16", np.float16)):
        path = args.out / f"embeddings_{name}.npy"
        np.save(path, embeddings.astype(dtype, copy=False))
        sizes[name] = path.stat().st_size

    manifest = {
        "checkpoint": str(args.checkpoint),
        "corpus": str(args.corpus or args.data / "corpus.parquet"),
        "commit": current_commit(),
        "documents": int(len(answer_ids)),
        "dimensions": int(embeddings.shape[1]),
        "max_length": args.max_length,
        "encode_seconds": round(seconds, 1),
        "device": retriever.device,
        "bytes": sizes,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(json.dumps(manifest, indent=2, sort_keys=True))
    for name, size in sizes.items():
        print(f"{name}: {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
