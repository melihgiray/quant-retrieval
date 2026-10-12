"""Download the model and retrieval files used by the hosted demo.

    python scripts/download_demo_assets.py melihgiray/quant-retrieval-model

The repository is expected to keep the transformer files at its root and the
corpus plus exported index under ``demo/``. Files are checked after download so
a partial upload fails during the build instead of during the first request.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from huggingface_hub import snapshot_download
from scripts.hub_auth import nonempty_hub_value, read_token_file

from quant_retrieval.eval.results import write_result
from quant_retrieval.serve.artifacts import (
    ALLOWED_FILES,
    CHECKSUM_FILE,
    file_sha256,
    verify_snapshot,
)

RECEIPT_FILE = ".download-receipt.json"


def download_demo_assets(
    repo_id: str, output: Path, revision: str = "main", *, token: str | None = None
) -> Path:
    repo_id = nonempty_hub_value(repo_id, "repo_id")
    revision = nonempty_hub_value(revision, "revision")
    if output.exists() and not output.is_dir():
        raise ValueError(f"asset output must be a directory: {output}")
    if output.is_symlink() or (output.exists() and any(output.iterdir())):
        raise FileExistsError("asset output must be a new or empty unlinked directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=f".{output.name}.") as temporary:
        staging = Path(temporary) / "assets"
        downloaded = Path(snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=staging,
            allow_patterns=list(ALLOWED_FILES),
            token=token,
        ))
        if downloaded.resolve() != staging.resolve():
            raise RuntimeError("asset download returned an unexpected directory")
        verify_snapshot(staging)
        write_result({
            "schema_version": 1,
            "repo_id": repo_id,
            "requested_revision": revision,
            "checksums_sha256": file_sha256(staging / CHECKSUM_FILE),
        }, staging / RECEIPT_FILE, overwrite=False)
        if output.is_symlink() or (output.exists() and any(output.iterdir())):
            raise FileExistsError("asset output changed during download")
        staging.rename(output)
    return output


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
