"""Small credential helpers shared by the Hub commands."""

from pathlib import Path


def read_token_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"token file does not exist: {path}")
    token = path.read_text().strip()
    if not token:
        raise RuntimeError(f"token file is empty: {path}")
    return token
