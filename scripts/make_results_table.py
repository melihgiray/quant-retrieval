"""Generate RESULTS.md from committed evaluation JSON files.

    python scripts/make_results_table.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def display_name(result: dict[str, Any]) -> str:
    configured = result["config"].get("display_name")
    if configured:
        return configured
    if result["retriever"] == "bm25":
        return "BM25"
    model_name = result["config"].get("parameters", {}).get("model_name", "dense")
    if model_name.endswith("all-MiniLM-L6-v2"):
        return "Frozen MiniLM"
    return result["run_name"].replace("_", " ").title()


def load_evaluations(results_dir: Path, split: str) -> list[dict[str, Any]]:
    evaluations = []
    for path in sorted(results_dir.glob("*.json")):
        result = json.loads(path.read_text())
        if result.get("split") == split and "metrics" in result:
            evaluations.append(result)
    return evaluations


def render_results(evaluations: list[dict[str, Any]], split: str) -> str:
    if not evaluations:
        raise ValueError(f"no {split} evaluation results found")

    lines = [
        "# Results",
        "",
        "These numbers come from the committed JSON files in `results/`. The evaluation ranks",
        "all 26,152 answers for each query. Validation is used until the final model is fixed.",
        "The test split has not been run.",
        "",
        "## Validation baselines",
        "",
        "| Retriever | nDCG@10 | MRR@10 | Recall@10 | Recall@100 "
        "| Graded nDCG@10 | Index time (s) | Search p50 (ms) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in evaluations:
        metrics = result["metrics"]
        timing = result["timing"]
        lines.append(
            f"| {display_name(result)} "
            f"| {metrics['ndcg_at_10']:.4f} "
            f"| {metrics['mrr_at_10']:.4f} "
            f"| {metrics['recall_at_10']:.4f} "
            f"| {metrics['recall_at_100']:.4f} "
            f"| {metrics['graded_ndcg_at_10']:.4f} "
            f"| {timing['index_seconds']:.2f} "
            f"| {timing['search_ms_per_query_p50']:.2f} |"
        )

    lines.extend(
        [
            "",
            "Strict metrics count only the accepted answer. Graded nDCG also gives partial",
            "credit to other nonnegative answers written for the same question.",
            "",
            "BM25 parameters are k1 1.2 and b 0.75. Frozen MiniLM is",
            "`sentence-transformers/all-MiniLM-L6-v2`, used without domain training. It mean",
            "pools nonpadding tokens, normalizes each 384 dimensional embedding, and ranks by",
            "cosine similarity.",
            "",
            "Timing was measured on an Apple M5 Pro. Index time includes tokenization and",
            "embedding for dense retrieval. Search latency includes query encoding and exact",
            "scoring against the full corpus. These are baseline measurements, not the final",
            "latency study.",
            "",
            "## Reproduce",
            "",
            "```sh",
            "python scripts/evaluate.py --config configs/bm25.yaml",
            "python scripts/evaluate.py --config configs/minilm_frozen.yaml",
            "python scripts/make_results_table.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("RESULTS.md"))
    parser.add_argument("--split", default="val")
    args = parser.parse_args()

    evaluations = load_evaluations(args.results, args.split)
    args.output.write_text(render_results(evaluations, args.split))
    print(f"wrote {args.output} from {len(evaluations)} evaluations")


if __name__ == "__main__":
    main()
