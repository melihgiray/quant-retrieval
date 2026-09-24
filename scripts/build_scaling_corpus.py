"""Build larger corpora so the ANN study has something to scale against.

    python scripts/build_scaling_corpus.py --sizes 100000 400000

Approximate search is pointless at 26,152 documents, so the question worth
answering is how large a corpus has to get before it pays. That needs corpora
larger than the one we have, and the cheapest honest way to get them is to pull
answers from other Stack Exchange sites and use them as distractors.

The quant answers are always included and always first, so every gold answer
stays reachable at every size. The queries and the judgements never change. Only
the haystack grows, which is exactly the variable under study.

Sizes nest: the 100k corpus is a prefix of the 400k one. Without that, a change
between two sizes could be the size or could be a different sample, and there
would be no way to tell which.
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from quant_retrieval.data.download import download_dump, extract_dump
from quant_retrieval.data.pairs import build_corpus
from quant_retrieval.data.parse import parse_posts
from quant_retrieval.eval.results import write_result

ARCHIVE = "https://archive.org/download/stackexchange"
# Similar in shape to quant.stackexchange: technical questions, long answers with
# mathematics, and the same markup. A corpus of unrelated prose would make the
# task artificially easy as it grew, since distractors nothing like the queries
# are not distractors.
#
# math.stackexchange would add the most documents and is deliberately left out.
# Its dump is 3.5GB, which does not fit a free Colab session alongside the
# download, the unpack and the embedding. stats and physics together are about
# 1.3GB and reach roughly 400k answers, which is enough to show where the curves
# cross. Add math if a machine with room turns up.
DEFAULT_SITES = ("stats.stackexchange.com", "physics.stackexchange.com")
ID_BLOCK_SIZE = 10_000_000


def validate_site(site: str) -> str:
    if not re.fullmatch(r"(?:[a-z0-9]+(?:-[a-z0-9]+)*\.)+stackexchange\.com", site):
        raise ValueError("site must be a lowercase Stack Exchange hostname")
    return site


def load_site(site: str, raw_root: Path) -> pd.DataFrame:
    """Download, unpack and clean one site's answers."""
    validate_site(site)
    destination = raw_root / site
    archive = destination / f"{site}.7z"
    info = download_dump(archive, url=f"{ARCHIVE}/{site}.7z")
    print(f"{site}: {info.bytes_downloaded / 1e6:.0f} MB, dated {info.last_modified}")
    extract_dump(archive, destination)

    _, answers = parse_posts(destination / "Posts.xml")
    corpus = build_corpus(answers)
    # Ids collide across sites, so give every site its own block. The study only
    # needs these documents to be distinct and retrievable, never to be looked up
    # in the original site.
    corpus = namespace_corpus(corpus, site)
    print(f"{site}: {len(corpus)} answers")
    return corpus


def _id_offset(site: str) -> int:
    """A stable, per site block of ids well clear of the quant corpus.

    crc32 rather than hash(), because Python randomises string hashing per
    process, so hash() would hand the same site a different id block on every
    run and the corpora would stop being reproducible.
    """
    return (zlib.crc32(site.encode()) % 900 + 100) * ID_BLOCK_SIZE


def namespace_corpus(corpus: pd.DataFrame, site: str) -> pd.DataFrame:
    result = corpus.copy()
    for column in ("answer_id", "question_id"):
        values = result[column]
        if (not pd.api.types.is_integer_dtype(values.dtype) or values.isna().any()
                or not values.between(1, ID_BLOCK_SIZE - 1).all()):
            raise ValueError(f"{site}: {column} exceeds its reserved ID block")
        result[column] = values.astype(np.int64) + _id_offset(site)
    return result


def combine_sources(base: pd.DataFrame, extras: list[pd.DataFrame]) -> pd.DataFrame:
    pool = pd.concat(extras, ignore_index=True) if extras else base.iloc[:0].copy()
    combined_ids = pd.concat([base["answer_id"], pool["answer_id"]])
    if combined_ids.duplicated().any():
        raise ValueError("answer IDs collide within or between scaling sources")
    return pool


def nested_corpora(base: pd.DataFrame, pool: pd.DataFrame, sizes: list[int], seed: int):
    """Yield deterministic nested samples, preserving every base answer in order."""
    if base.empty or not sizes:
        raise ValueError("base corpus and requested sizes must not be empty")
    capacity = len(base) + len(pool)
    if any(size < len(base) or size > capacity for size in sizes):
        raise ValueError(f"all corpus sizes must be between {len(base)} and {capacity}")
    pool = pool.sort_values("answer_id").reset_index(drop=True)
    order = np.random.default_rng(seed).permutation(len(pool))
    shuffled = pool.iloc[order]
    for size in sorted(set(sizes)):
        yield size, pd.concat([base, shuffled.head(size - len(base))], ignore_index=True)


def write_corpus(corpus: pd.DataFrame, path: Path) -> None:
    """Replace a corpus only after its complete parquet payload is on disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as file:
        temporary = Path(file.name)
    try:
        corpus.to_parquet(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw/scaling"))
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    parser.add_argument("--sites", nargs="+", type=validate_site, default=list(DEFAULT_SITES))
    parser.add_argument("--sizes", nargs="+", type=int, default=[100_000, 400_000])
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if any(size <= 0 for size in args.sizes):
        parser.error("corpus sizes must be positive")
    if len(set(args.sites)) != len(args.sites):
        parser.error("sites must not be repeated")
    args.sizes = sorted(set(args.sizes))
    if len({_id_offset(site) for site in args.sites}) != len(args.sites):
        parser.error("sites map to colliding ID blocks; choose different source sites")

    base = pd.read_parquet(args.data / "corpus.parquet")
    print(f"quant corpus: {len(base)} answers, always kept")

    extras = [load_site(site, args.raw) for site in args.sites]
    pool = combine_sources(base, extras)

    args.out.mkdir(parents=True, exist_ok=True)
    summary = {"base_documents": int(len(base)), "pool_documents": int(len(pool)), "corpora": {}}

    for size, corpus in nested_corpora(base, pool, args.sizes, args.seed):
        path = args.out / f"scaling_corpus_{size}.parquet"
        write_corpus(corpus, path)
        summary["corpora"][str(len(corpus))] = str(path)
        print(f"wrote {path} with {len(corpus)} documents")

    write_result(summary, args.out / "scaling_corpora.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
