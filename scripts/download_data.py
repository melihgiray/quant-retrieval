"""Download and unpack the Stack Exchange dump.

    python scripts/download_data.py

Writes data/raw/quant.stackexchange.com.7z, the extracted XML next to it, and
data/raw/dump_info.json with the dump date to cite in docs/DATA.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from quant_retrieval.data.download import DUMP_URL, download_dump, extract_dump


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    parser.add_argument("--url", default=DUMP_URL)
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    archive = args.out / "quant.stackexchange.com.7z"
    info = download_dump(archive, url=args.url, force=args.force)
    info.to_json(args.out / "dump_info.json")
    print(f"archive: {archive} ({info.bytes_downloaded / 1e6:.1f} MB)")
    print(f"dump date (Last-Modified): {info.last_modified}")

    for path in extract_dump(archive, args.out):
        print(f"extracted: {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
