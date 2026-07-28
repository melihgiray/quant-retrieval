"""Fetch and unpack the quant.stackexchange.com data dump.

The dump is a 7z archive of XML files, one per table, published by Stack
Exchange on archive.org. It is about 55MB compressed. We keep the download
and the extraction separate so a failed extraction does not mean fetching
55MB again.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import py7zr
from tqdm import tqdm

DUMP_URL = "https://archive.org/download/stackexchange/quant.stackexchange.com.7z"

# The only tables we need. Posts holds questions and answers, PostLinks holds
# the duplicate and related edges we use to keep near identical questions from
# landing on both sides of a split, and Tags carries the site's tag counts for
# the dataset write-up.
WANTED_MEMBERS = ("Posts.xml", "PostLinks.xml", "Tags.xml")


@dataclass(frozen=True)
class DumpInfo:
    """What we know about the archive we actually downloaded."""

    url: str
    bytes_downloaded: int
    last_modified: str | None

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.__dict__, indent=2) + "\n")


def download_dump(dest: Path, url: str = DUMP_URL, force: bool = False) -> DumpInfo:
    """Download the archive to `dest`, skipping the transfer if it is already there.

    Returns the archive metadata, including the server's Last-Modified date. That
    date is the dump date and belongs in docs/DATA.md, since Stack Exchange
    republishes these quarterly and the counts move.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    request = urllib.request.Request(url, headers={"User-Agent": "quant-retrieval/0.1"})
    with urllib.request.urlopen(request) as response:  # noqa: S310 (fixed https URL)
        expected = int(response.headers.get("Content-Length", 0))
        last_modified = response.headers.get("Last-Modified")

        if dest.exists() and not force and dest.stat().st_size == expected:
            return DumpInfo(url=url, bytes_downloaded=expected, last_modified=last_modified)

        written = 0
        with (
            open(dest, "wb") as out,
            tqdm(total=expected or None, unit="B", unit_scale=True, desc=dest.name) as bar,
        ):
            while chunk := response.read(1 << 20):
                out.write(chunk)
                written += len(chunk)
                bar.update(len(chunk))

    if expected and written != expected:
        dest.unlink(missing_ok=True)
        raise OSError(f"download truncated: got {written} bytes, expected {expected}")

    return DumpInfo(url=url, bytes_downloaded=written, last_modified=last_modified)


def extract_dump(
    archive: Path, dest_dir: Path, members: tuple[str, ...] = WANTED_MEMBERS
) -> list[Path]:
    """Extract the wanted XML files. Returns the paths that now exist on disk."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    with py7zr.SevenZipFile(archive, mode="r") as zf:
        present = set(zf.getnames())
        missing = [name for name in members if name not in present]
        if missing:
            raise KeyError(f"archive is missing {missing}, it holds {sorted(present)}")
        zf.extract(path=dest_dir, targets=list(members))
    return [dest_dir / name for name in members]
