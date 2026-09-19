"""Small credential helpers shared by the Hub commands."""

from pathlib import Path


def nonempty_hub_value(value: str, name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


def read_token_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"token file does not exist: {path}")
    token = path.read_text().strip()
    if not token:
        raise RuntimeError(f"token file is empty: {path}")
    return token
