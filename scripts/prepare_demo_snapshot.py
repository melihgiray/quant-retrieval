"""Assemble the trained model and retrieval files into one upload directory."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from quant_retrieval.serve.artifacts import (
    CHECKSUM_FILE,
    MODEL_FILES,
    RETRIEVAL_FILES,
    linked_paths,
    write_checksums,
)


def prepare_demo_snapshot(
    checkpoint: Path,
    corpus: Path,
    artifacts: Path,
    output: Path,
) -> Path:
    if output.exists() and not output.is_dir():
        raise ValueError(f"snapshot output must be a directory: {output}")
    resolved_output = output.resolve()
    nested_sources = [
        str(source_root)
        for source_root in (checkpoint.resolve(), artifacts.resolve())
        if resolved_output != source_root and resolved_output.is_relative_to(source_root)
    ]
    if nested_sources:
        raise ValueError(f"snapshot output is nested inside source directories: {nested_sources}")
    sources = {name: checkpoint / name for name in MODEL_FILES}
    sources.update(
        {
            f"demo/{name}": corpus if name == "corpus.parquet" else artifacts / name
            for name in RETRIEVAL_FILES
        }
    )
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"cannot prepare snapshot, missing files: {missing}")
    linked_sources = [relative for relative, source in sources.items() if source.is_symlink()]
    if linked_sources:
        raise ValueError(f"snapshot sources contain linked files: {linked_sources}")
    collisions = [
        relative
        for relative, source in sources.items()
        if (output / relative).resolve() == source.resolve()
    ]
    if collisions:
        raise ValueError(f"snapshot output overlaps source files: {collisions}")
    linked = linked_paths(output, tuple(sources))
    if linked:
        raise ValueError(f"snapshot output contains linked paths: {linked}")

    (output / CHECKSUM_FILE).unlink(missing_ok=True)
    for relative, source in sources.items():
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    write_checksums(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/minilm_tuned/epoch-3")
    )
    parser.add_argument("--corpus", type=Path, default=Path("data/processed/corpus.parquet"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("demo_assets"))
    args = parser.parse_args()
    output = prepare_demo_snapshot(args.checkpoint, args.corpus, args.artifacts, args.output)
    print(f"prepared demo snapshot at {output}")


if __name__ == "__main__":
    main()
