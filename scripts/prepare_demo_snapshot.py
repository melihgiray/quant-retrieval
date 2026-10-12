"""Assemble the trained model and retrieval files into one upload directory."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from quant_retrieval.retrieval.checkpoint import TOKENIZER_EXTRAS, verify_checkpoint
from quant_retrieval.retrieval.index_artifacts import read_manifest
from quant_retrieval.serve.artifacts import (
    MODEL_FILES,
    RETRIEVAL_FILES,
    file_sha256,
    linked_paths,
    write_checksums,
)
from quant_retrieval.serve.snapshot_validation import (
    inspect_snapshot,
    validate_snapshot_corpus,
    validate_snapshot_vectors,
)


def prepare_demo_snapshot(
    checkpoint: Path,
    corpus: Path,
    artifacts: Path,
    output: Path,
) -> Path:
    if checkpoint.is_symlink() or artifacts.is_symlink():
        raise ValueError("snapshot sources contain linked directories")
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
    sources.update({name: checkpoint / name for name in TOKENIZER_EXTRAS
                    if (checkpoint / name).exists() or (checkpoint / name).is_symlink()})
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
    empty_sources = [relative for relative, source in sources.items() if source.stat().st_size == 0]
    if empty_sources:
        raise ValueError(f"snapshot sources contain empty files: {empty_sources}")
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
    if output.exists():
        raise FileExistsError("snapshot output already exists; choose a new version directory")

    source_hashes = {relative: file_sha256(source) for relative, source in sources.items()}
    manifest = read_manifest(artifacts / "manifest.json")
    verify_checkpoint(checkpoint, manifest)
    frame = validate_snapshot_corpus(corpus, manifest)
    validate_snapshot_vectors(artifacts, manifest, frame)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=f".{output.name}.") as temporary:
        staging = Path(temporary) / "snapshot"
        for relative, source in sources.items():
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            if file_sha256(destination) != source_hashes[relative]:
                raise ValueError(f"snapshot source changed during preparation: {relative}")
        write_checksums(staging)
        inspect_snapshot(staging)
        if output.exists() or output.is_symlink():
            raise FileExistsError("snapshot output appeared during preparation")
        staging.rename(output)
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
