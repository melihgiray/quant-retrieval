"""Download the model and retrieval files used by the hosted demo.

    python scripts/download_demo_assets.py melihgiray/quant-retrieval-model

The repository is expected to keep the transformer files at its root and the
corpus plus exported index under ``demo/``. Files are checked after download so
a partial upload fails during the build instead of during the first request.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download
from scripts.hub_auth import read_token_file

from quant_retrieval.serve.artifacts import REQUIRED_FILES, verify_snapshot


def download_demo_assets(
    repo_id: str, output: Path, revision: str = "main", *, token: str | None = None
) -> Path:
    downloaded = Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=output,
            allow_patterns=list(REQUIRED_FILES),
            token=token,
        )
    )
    verify_snapshot(downloaded)
    return downloaded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output", type=Path, default=Path("demo_assets"))
    parser.add_argument("--token-file", type=Path)
    args = parser.parse_args()
    token = read_token_file(args.token_file) if args.token_file else None
    path = download_demo_assets(args.repo_id, args.output, args.revision, token=token)
    print(f"downloaded demo assets to {path}")


if __name__ == "__main__":
    main()
