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
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from quant_retrieval.data.download import download_dump, extract_dump
from quant_retrieval.data.pairs import build_corpus
from quant_retrieval.data.parse import parse_posts

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


def load_site(site: str, raw_root: Path) -> pd.DataFrame:
    """Download, unpack and clean one site's answers."""
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
    corpus["answer_id"] = corpus["answer_id"] + _id_offset(site)
    corpus["question_id"] = corpus["question_id"] + _id_offset(site)
    print(f"{site}: {len(corpus)} answers")
    return corpus


def _id_offset(site: str) -> int:
    """A stable, per site block of ids well clear of the quant corpus.

    crc32 rather than hash(), because Python randomises string hashing per
    process, so hash() would hand the same site a different id block on every
    run and the corpora would stop being reproducible.
    """
    return (zlib.crc32(site.encode()) % 900 + 100) * 10_000_000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw/scaling"))
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    parser.add_argument("--sites", nargs="+", default=list(DEFAULT_SITES))
    parser.add_argument("--sizes", nargs="+", type=int, default=[100_000, 400_000])
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    base = pd.read_parquet(args.data / "corpus.parquet")
    print(f"quant corpus: {len(base)} answers, always kept")

    extras = [load_site(site, args.raw) for site in args.sites]
    pool = pd.concat(extras, ignore_index=True) if extras else pd.DataFrame(columns=base.columns)
    pool = pool[~pool["answer_id"].isin(set(base["answer_id"]))]

    # One shuffle, then prefixes of it. That is what makes the sizes nest.
    order = np.random.default_rng(args.seed).permutation(len(pool))
    pool = pool.iloc[order].reset_index(drop=True)

    args.out.mkdir(parents=True, exist_ok=True)
    summary = {"base_documents": int(len(base)), "pool_documents": int(len(pool)), "corpora": {}}

    for size in sorted(args.sizes):
        wanted = size - len(base)
        if wanted < 0:
            print(f"skipping {size}, smaller than the quant corpus alone")
            continue
        if wanted > len(pool):
            print(f"skipping {size}, only {len(base) + len(pool)} documents available")
            continue
        corpus = pd.concat([base, pool.head(wanted)], ignore_index=True)
        path = args.out / f"scaling_corpus_{len(corpus)}.parquet"
        corpus.to_parquet(path, index=False)
        summary["corpora"][str(len(corpus))] = str(path)
        print(f"wrote {path} with {len(corpus)} documents")

    (args.out / "scaling_corpora.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
