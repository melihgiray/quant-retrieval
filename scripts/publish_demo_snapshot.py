"""Publish a prepared and verified demo snapshot to the Hub."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi
from scripts.hub_auth import nonempty_hub_value, read_token_file

from quant_retrieval.serve.artifacts import REQUIRED_FILES, verify_snapshot


def publish_demo_snapshot(
    repo_id: str,
    snapshot: Path,
    *,
    private: bool = True,
    token: str | None = None,
    api: Any | None = None,
) -> str:
    repo_id = nonempty_hub_value(repo_id, "repo_id")
    verify_snapshot(snapshot)
    client = api or HfApi(token=token)
    client.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    repository = client.repo_info(repo_id=repo_id, repo_type="model")
    if repository.private != private:
        requested = "private" if private else "public"
        actual = "private" if repository.private else "public"
        raise RuntimeError(
            f"asset repository is {actual}, but {requested} visibility was requested; "
            "change the repository setting or choose the matching publication option"
        )
    result = client.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(snapshot),
        allow_patterns=list(REQUIRED_FILES),
        commit_message="Publish search demo snapshot",
    )
    return str(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id")
    parser.add_argument("--snapshot", type=Path, default=Path("demo_assets"))
    parser.add_argument("--token-file", type=Path, default=Path("~/.hf_token").expanduser())
    parser.add_argument("--public", action="store_true")
    args = parser.parse_args()

    token = read_token_file(args.token_file)
    url = publish_demo_snapshot(
        args.repo_id,
        args.snapshot,
        private=not args.public,
        token=token,
    )
    print(f"published demo snapshot: {url}")


if __name__ == "__main__":
    main()
