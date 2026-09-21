"""Check readiness and one search against a running demo."""

from __future__ import annotations

import argparse
import json
import math
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode, urlsplit

from quant_retrieval.eval.results import write_result


def _finite_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _get_json(url: str, timeout: float) -> dict:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        body = response.read(1_000_001)
    if len(body) > 1_000_000:
        raise ValueError("demo response exceeds one megabyte")
    payload = json.loads(body)
    if not isinstance(payload, dict):
        raise ValueError("demo response must be a JSON object")
    return payload


def check_demo(
    base_url: str, *, query: str = "How do I calculate implied volatility?",
    k: int = 3, timeout: float = 30.0, expected_commit: str | None = None,
) -> dict:
    """Fail unless the expected pipeline is ready and returns a usable ranking."""
    parsed = urlsplit(base_url)
    if (
        parsed.scheme not in {"http", "https"} or not parsed.hostname
        or parsed.username is not None or parsed.password is not None
        or parsed.query or parsed.fragment
    ):
        raise ValueError("base URL must be HTTP(S), without credentials, query or fragment")
    if not query.strip() or len(query) > 1000:
        raise ValueError("query must contain 1 to 1000 characters")
    if type(k) is not int or not 1 <= k <= 20:
        raise ValueError("k must be an integer from 1 to 20")
    if not _finite_number(timeout) or timeout <= 0:
        raise ValueError("timeout must be positive and finite")
    base_url = base_url.rstrip("/")
    health = _get_json(f"{base_url}/health", timeout)
    if (
        health.get("ready") is not True or health.get("pipeline") != "bm25_dense_rrf"
        or type(health.get("documents")) is not int or health["documents"] < 1
    ):
        raise ValueError("demo is not ready with a populated hybrid pipeline")
    if expected_commit is not None and health.get("artifact_commit") != expected_commit:
        raise ValueError("loaded artifact commit does not match the expected revision")

    parameters = urlencode({"q": query.strip(), "k": k})
    search = _get_json(f"{base_url}/search?{parameters}", timeout)
    if search.get("query") != query.strip():
        raise ValueError("search response does not match the requested query")
    elapsed = search.get("elapsed_ms")
    if not _finite_number(elapsed) or elapsed < 0:
        raise ValueError("search response has invalid elapsed time")
    hits = search.get("results")
    if not isinstance(hits, list) or not 1 <= len(hits) <= min(k, health["documents"]):
        raise ValueError("search must return a nonempty ranking within the requested limit")
    seen = set()
    previous = math.inf
    for hit in hits:
        if not isinstance(hit, dict):
            raise ValueError("search hit must be an object")
        answer_id = hit.get("answer_id")
        question_id = hit.get("question_id")
        score = hit.get("score")
        if (
            type(answer_id) is not int or answer_id < 1 or answer_id in seen
            or type(question_id) is not int or question_id < 1
            or not _finite_number(score) or score > previous
        ):
            raise ValueError("search ranking contains invalid IDs or scores")
        if not isinstance(hit.get("text"), str) or not hit["text"].strip():
            raise ValueError("search hit has no answer text")
        if hit.get("url") != f"https://quant.stackexchange.com/a/{answer_id}":
            raise ValueError("search hit has an unexpected answer URL")
        seen.add(answer_id)
        previous = score
    return {
        "checked_at": datetime.now(UTC).isoformat(),
        "request": {"base_url": base_url, "query": query.strip(), "k": k,
                    "timeout": timeout, "expected_commit": expected_commit},
        "health": health,
        "search": search,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url")
    parser.add_argument("--query", default="How do I calculate implied volatility?")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--expected-commit")
    parser.add_argument("--out", type=Path, help="save a successful check as an atomic JSON report")
    args = parser.parse_args()
    try:
        report = check_demo(
            args.base_url, query=args.query, k=args.k, timeout=args.timeout,
            expected_commit=args.expected_commit,
        )
        if args.out is not None:
            write_result(report, args.out)
    except (OSError, ValueError) as error:
        raise SystemExit(f"demo check failed: {error}") from error
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
