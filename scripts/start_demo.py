"""Start the search demo with local files or a remote asset snapshot."""

from __future__ import annotations

import argparse
import os
from collections.abc import MutableMapping
from pathlib import Path

import uvicorn
from scripts.download_demo_assets import download_demo_assets


def configure_asset_environment(
    root: Path, environ: MutableMapping[str, str] | None = None
) -> None:
    values = os.environ if environ is None else environ
    defaults = {
        "MODEL_PATH": root,
        "CORPUS_PATH": root / "demo/corpus.parquet",
        "MANIFEST_PATH": root / "demo/manifest.json",
        "DOCUMENT_IDS_PATH": root / "demo/answer_ids.npy",
        "EMBEDDINGS_PATH": root / "demo/embeddings_fp16.npy",
    }
    for name, path in defaults.items():
        values.setdefault(name, str(path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "7860")))
    parser.add_argument("--asset-repo", default=os.getenv("ASSET_REPO"))
    parser.add_argument("--asset-revision", default=os.getenv("ASSET_REVISION", "main"))
    parser.add_argument("--asset-dir", type=Path, default=Path("demo_assets"))
    args = parser.parse_args()

    if args.asset_repo:
        root = download_demo_assets(args.asset_repo, args.asset_dir, args.asset_revision)
        configure_asset_environment(root)
    uvicorn.run("quant_retrieval.serve.app:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
