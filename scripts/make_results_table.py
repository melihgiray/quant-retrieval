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


def improvement_sentence(evaluations: list[dict[str, Any]]) -> list[str]:
    """State the fine-tuning gain, computed from the results rather than typed."""
    frozen = next((r for r in evaluations if r["run_name"] == "minilm_frozen_val"), None)
    tuned = [r for r in evaluations if r["run_name"].startswith("minilm_tuned")]
    if frozen is None or not tuned:
        return []

    best = max(tuned, key=lambda r: r["metrics"]["ndcg_at_10"])
    before = frozen["metrics"]["ndcg_at_10"]
    after = best["metrics"]["ndcg_at_10"]
    return [
        f"Fine-tuning moves nDCG@10 from {before:.4f} to {after:.4f}, {after - before:+.4f}",
        f"absolute and {(after - before) / before:+.1%} relative against the same encoder",
        "untrained. Recall@100 is the number to watch for the reranking stage later, since",
        "nothing a reranker does can recover an answer that never made the candidate list.",
        "",
    ]


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
        "## Validation results",
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

    lines.append("")
    lines.extend(improvement_sentence(evaluations))
    lines.extend(
        [
            "Strict metrics count only the accepted answer. Graded nDCG also gives partial",
            "credit to other nonnegative answers written for the same question.",
            "",
            "## What each row is",
            "",
            "BM25 parameters are k1 1.2 and b 0.75. Frozen MiniLM is",
            "`sentence-transformers/all-MiniLM-L6-v2`, used without domain training. It mean",
            "pools nonpadding tokens, normalizes each 384 dimensional embedding, and ranks by",
            "cosine similarity.",
            "",
            "The tuned rows are that same encoder fine-tuned on the 9,924 training pairs with",
            "an in-batch contrastive loss: each question is pulled toward its own answer and",
            "pushed away from the other 63 answers in its batch. Three epochs, batch size 64,",
            "AdamW at 2e-5 with linear warmup over the first tenth of steps and linear decay",
            "after, gradient clipping at 1.0, temperature 0.05. Training took about eight",
            "minutes on an Apple M5 Pro.",
            "",
            "All three epoch checkpoints are listed because the checkpoint was chosen on",
            "validation, and showing the choice is more useful than asserting it. The gain",
            "flattens between epoch 2 and epoch 3, and epoch 2 is actually the better",
            "checkpoint on Recall@100, so the ranking depends on which metric is being served.",
            "",
            "Timing was measured on an Apple M5 Pro. Index time includes tokenization and",
            "embedding for dense retrieval. Search latency includes query encoding and exact",
            "scoring against the full corpus. These are not the final latency study.",
            "",
            "## Reproduce",
            "",
            "```sh",
            "python scripts/evaluate.py --config configs/bm25.yaml",
            "python scripts/evaluate.py --config configs/minilm_frozen.yaml",
            "python scripts/train.py --config configs/base.yaml",
            "python scripts/evaluate.py --config configs/minilm_tuned_epoch3.yaml",
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
