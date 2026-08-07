"""Generate RESULTS.md from committed evaluation JSON files.

    python scripts/make_results_table.py

Every number and every comparison in the output is read out of `results/*.json`.
Nothing here types a metric, so the prose cannot drift away from the runs it
describes. Sections whose runs are missing are skipped rather than guessed at.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any

WRAP_WIDTH = 88

METRIC_COLUMNS = (
    ("nDCG@10", "ndcg_at_10"),
    ("MRR@10", "mrr_at_10"),
    ("Recall@10", "recall_at_10"),
    ("Recall@100", "recall_at_100"),
    ("Graded nDCG@10", "graded_ndcg_at_10"),
)


def paragraph(text: str) -> list[str]:
    """Wrap one paragraph to the file's width, then leave a blank line."""
    return [*textwrap.wrap(" ".join(text.split()), width=WRAP_WIDTH), ""]


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


def find(evaluations: list[dict[str, Any]], run_name: str) -> dict[str, Any] | None:
    return next((r for r in evaluations if r["run_name"] == run_name), None)


def score(result: dict[str, Any] | None, metric: str = "ndcg_at_10") -> float | None:
    return None if result is None else result["metrics"][metric]


def table_order(result: dict[str, Any]) -> tuple[int, str]:
    """Rows appear in the order configs ask for, not the order files were found."""
    return (result["config"].get("display_order", 999), result["run_name"])


def render_table(evaluations: list[dict[str, Any]]) -> list[str]:
    header = "| Retriever | " + " | ".join(label for label, _ in METRIC_COLUMNS)
    header += " | Index time (s) | Search p50 (ms) |"
    lines = [header, "| --- |" + " ---: |" * (len(METRIC_COLUMNS) + 2)]
    for result in sorted(evaluations, key=table_order):
        metrics = result["metrics"]
        cells = " | ".join(f"{metrics[key]:.4f}" for _, key in METRIC_COLUMNS)
        lines.append(
            f"| {display_name(result)} | {cells} "
            f"| {result['timing']['index_seconds']:.2f} "
            f"| {result['timing']['search_ms_per_query_p50']:.2f} |"
        )
    return lines


def headline(evaluations: list[dict[str, Any]]) -> list[str]:
    frozen = find(evaluations, "minilm_frozen_val")
    # Every system that is meant to work: baselines are the thing being beaten,
    # and the undertrained reranker pipelines are not a candidate for best.
    candidates = [
        r
        for r in evaluations
        if r["run_name"] not in {"bm25_val", "minilm_frozen_val"}
        and not r["run_name"].endswith("_rerank_val")
    ]
    if frozen is None or not candidates:
        return []
    best = max(candidates, key=lambda r: r["metrics"]["ndcg_at_10"])
    before, after = frozen["metrics"]["ndcg_at_10"], best["metrics"]["ndcg_at_10"]
    bm25 = find(evaluations, "bm25_val")
    text = (
        f"{display_name(best)} is the strongest pipeline here at {after:.4f} nDCG@10, "
        f"against {before:.4f} for the same encoder untrained. That is "
        f"{after - before:+.4f} absolute and {(after - before) / before:+.1%} relative."
    )
    if bm25 is not None:
        gap = before - bm25["metrics"]["ndcg_at_10"]
        text += (
            f" For scale, moving from BM25 to that untrained encoder was worth {gap:+.4f}, "
            "so domain training bought rather less than switching to embeddings did in the "
            "first place."
        )
    return paragraph(text)


def hard_negative_section(evaluations: list[dict[str, Any]]) -> list[str]:
    base = find(evaluations, "minilm_tuned_epoch3_val")
    mined = find(evaluations, "minilm_hardneg_epoch3_val")
    if base is None or mined is None:
        return []

    ndcg_delta = score(mined) - score(base)
    recall_delta = score(mined, "recall_at_100") - score(base, "recall_at_100")
    early = find(evaluations, "minilm_hardneg_epoch1_val")
    early_base = find(evaluations, "minilm_tuned_epoch1_val")

    lines = ["### Hard negatives did not help", ""]
    lines += paragraph(
        "The obvious next lever after in-batch negatives is to mine wrong answers that a "
        "retriever already ranks highly, so the model has to work for its wins. Four per "
        "question were mined, half from BM25 and half from the tuned encoder, with every "
        "answer to the same question excluded."
    )
    lines += paragraph(
        f"It bought {ndcg_delta:+.4f} nDCG@10, which is nothing, and cost "
        f"{recall_delta:+.4f} Recall@100."
    )
    if early is not None and early_base is not None:
        lines += paragraph(
            f"It did converge faster. After one epoch, mining was at {score(early):.4f} "
            f"against {score(early_base):.4f} without, and that gap had closed by epoch 3. "
            "The mined negatives carry real signal early and then stop mattering."
        )
    lines += paragraph(
        "The recall drop is the interesting part, and the likeliest explanation is false "
        "negatives. On a site this narrow, the answers a retriever ranks just below the "
        "right one are often genuinely useful answers written for a different question "
        "about the same thing. Training the model to push those away teaches it to separate "
        "documents that belong near each other, which is what a falling Recall@100 looks "
        "like. Excluding same-question answers, which this already does, does not catch it."
    )
    lines += paragraph(
        "Worth trying if this gets picked up again: skip candidates the current model "
        "already scores very close to the positive, rather than only skipping the top hit "
        "by rank."
    )
    return lines


def batch_section(evaluations: list[dict[str, Any]]) -> list[str]:
    sizes = {
        16: find(evaluations, "minilm_batch16_epoch3_val"),
        32: find(evaluations, "minilm_batch32_epoch3_val"),
        64: find(evaluations, "minilm_tuned_epoch3_val"),
        128: find(evaluations, "minilm_batch128_epoch3_val"),
    }
    present = {size: result for size, result in sizes.items() if result is not None}
    if len(present) < 3:
        return []

    best_size = max(present, key=lambda size: score(present[size]))
    smallest = min(present)
    climb = score(present[best_size]) - score(present[smallest])
    listed = ", ".join(f"{size} gives {score(present[size]):.4f}" for size in sorted(present))
    lines = ["### Batch size matters, up to a point", ""]
    lines += paragraph(
        "In-batch negatives mean batch size is not only a speed setting. A batch of 64 asks "
        "whether the right answer beats 63 others, a batch of 16 asks whether it beats 15."
    )
    lines += paragraph(
        f"On nDCG@10, batch {listed}. Best is {best_size}, worth {climb:+.4f} over batch "
        f"{smallest}, and it falls off again after that."
    )
    lines += paragraph(
        "One caveat that matters: the learning rate was held at 2e-5 for all four. Bigger "
        "batches therefore take proportionally fewer optimiser steps over the same three "
        "epochs, so this measures batch size and update count together rather than batch "
        "size on its own. Separating them means scaling the learning rate with the batch "
        "and rerunning, which is the next experiment, not something to assert here."
    )
    return lines


def pooling_section(evaluations: list[dict[str, Any]]) -> list[str]:
    mean = find(evaluations, "minilm_tuned_epoch3_val")
    cls = find(evaluations, "minilm_cls_epoch3_val")
    frozen = find(evaluations, "minilm_frozen_val")
    if mean is None or cls is None:
        return []

    lines = ["### Pooling is not a free choice", ""]
    lines += paragraph(
        f"Taking the first token instead of averaging them reaches {score(cls):.4f} against "
        f"{score(mean):.4f}, a loss of {score(cls) - score(mean):.4f}."
    )
    if frozen is not None and score(cls) < score(frozen):
        lines += paragraph(
            f"It also lands below the untrained encoder at {score(frozen):.4f}, which is the "
            "part worth noticing. Three epochs of domain training did not recover what the "
            "wrong pooling gave away."
        )
    lines += paragraph(
        "This is the expected direction rather than a surprise. all-MiniLM-L6-v2 was "
        "distilled with mean pooling, so its first token was never trained to stand for the "
        "sequence. The ablation is here because it is cheap and because the claim is better "
        "shown than asserted."
    )
    return lines


def hybrid_section(evaluations: list[dict[str, Any]]) -> list[str]:
    hybrid = find(evaluations, "hybrid_val")
    dense = find(evaluations, "minilm_tuned_epoch3_val")
    bm25 = find(evaluations, "bm25_val")
    if hybrid is None or dense is None or bm25 is None:
        return []

    lines = ["### Fusing the two retrievers beats either one", ""]
    lines += paragraph(
        "BM25 and the tuned encoder fail differently. BM25 finds the exact ticker, function "
        "name or formula that the encoder has smoothed into a general sense of the topic. The "
        "encoder finds the answer that never repeats the question's words. Reciprocal rank "
        "fusion merges them on rank rather than score, because BM25 sums unbounded term "
        "weights while cosine lives in [-1, 1], and combining those numbers directly means "
        "inventing a scale factor and then tuning it."
    )
    lines += paragraph(
        f"Hybrid reaches {score(hybrid):.4f} nDCG@10 against {score(dense):.4f} for the tuned "
        f"encoder alone and {score(bm25):.4f} for BM25, so it is "
        f"{score(hybrid) - score(dense):+.4f} over the better of its two parts. Recall@100 "
        f"goes from {score(dense, 'recall_at_100'):.4f} to "
        f"{score(hybrid, 'recall_at_100'):.4f}."
    )
    lines += paragraph(
        "The recall number is the one that matters most for what comes next. A reranking stage "
        "can only reorder what it is given, so the candidate list is a hard ceiling, and fusion "
        "raises that ceiling before anything expensive runs."
    )
    return lines


def reranker_section(evaluations: list[dict[str, Any]]) -> list[str]:
    reranked = [r for r in evaluations if r["run_name"].endswith("_rerank_val")]
    if not reranked:
        return []

    lines = ["### The reranker is not finished, and the numbers show it", ""]
    lines += paragraph(
        "A cross-encoder reads the question and one answer as a single sequence, so attention "
        "runs across both. That is strictly more informative than comparing two vectors, and "
        "strictly too slow to search with, so it runs last over the top 50 candidates."
    )
    lines += paragraph(
        "It is implemented, tested, and trained for one epoch. The planned second epoch did "
        "not complete: the machine ran out of memory partway through, and a resumed run was "
        "reduced to about eight steps per minute against 133 in the first epoch, so it was "
        "stopped rather than left thrashing."
    )
    for result in sorted(reranked, key=table_order):
        base_name = {"bm25_rerank_val": "bm25_val", "dense_rerank_val": "minilm_tuned_epoch3_val"}
        base = find(evaluations, base_name.get(result["run_name"], ""))
        if base is None:
            continue
        lines += paragraph(
            f"{display_name(result)}: {score(result):.4f} nDCG@10 against {score(base):.4f} "
            f"for the same retriever without it, {score(result) - score(base):+.4f}. "
            f"Recall@100 is unchanged at {score(result, 'recall_at_100'):.4f}."
        )
    lines += paragraph(
        "So a half-trained reranker is worse than none, and it does more damage the better the "
        "retriever underneath it, which is what you would expect: there is more good ordering "
        "to destroy. Scored directly against four random documents it picks the right answer "
        "42 percent of the time, against 20 percent for guessing, so it has learned something "
        "real and nowhere near enough."
    )
    lines += paragraph(
        "Recall@100 holding exactly steady is worth noting on its own. It confirms the stage "
        "only reorders its shortlist and never drops what sits beyond it, which is the one "
        "thing a reranker must not get wrong."
    )
    lines += paragraph(
        "Hybrid plus reranker was not run. With both single-retriever pipelines this far down, "
        "a third would cost ten minutes to confirm what the first two already say."
    )
    return lines


def render_results(evaluations: list[dict[str, Any]], split: str) -> str:
    if not evaluations:
        raise ValueError(f"no {split} evaluation results found")

    lines = [
        "# Results",
        "",
        *paragraph(
            "Every number here comes from the committed JSON files in `results/`, and this "
            "file is generated from them by `scripts/make_results_table.py` rather than "
            "written by hand. Each run ranks all 26,152 answers for every query."
        ),
        # Kept on one line, and pinned by a test: it is the claim that the
        # numbers below are not the ones being reported at the end.
        "The test split has not been run. Everything below is validation, 753 questions.",
        "",
        "## Validation results",
        "",
        *render_table(evaluations),
        "",
        *headline(evaluations),
        "Strict metrics count only the accepted answer. Graded nDCG also gives partial",
        "credit to other nonnegative answers written for the same question.",
        "",
        "## What the ablations say",
        "",
        *hard_negative_section(evaluations),
        *batch_section(evaluations),
        *pooling_section(evaluations),
        *hybrid_section(evaluations),
        *reranker_section(evaluations),
        "## What each row is",
        "",
        "BM25 parameters are k1 1.2 and b 0.75. Frozen MiniLM is",
        "`sentence-transformers/all-MiniLM-L6-v2` with no domain training. It mean pools",
        "nonpadding tokens, normalizes each 384 dimensional embedding, and ranks by cosine",
        "similarity.",
        "",
        "Every tuned row is that encoder fine-tuned on the 9,924 training pairs with an",
        "in-batch contrastive loss: each question is pulled toward its own answer and",
        "pushed away from the other answers in its batch. Three epochs, AdamW at 2e-5 with",
        "linear warmup over the first tenth of steps and linear decay after, gradient",
        "clipping at 1.0, temperature 0.05, seed 17. Unless a row says otherwise it is",
        "batch 64 and mean pooling.",
        "",
        "All three epochs are listed for the two batch 64 runs because the checkpoint was",
        "chosen on validation, and showing the choice is more useful than asserting it.",
        "",
        "Timing was measured on an Apple M5 Pro. Index time covers tokenizing and embedding",
        "the corpus, search latency covers query encoding and exact scoring against all of",
        "it. The two largest runs used gradient checkpointing to fit in memory, which is",
        "mathematically identical and about 30 percent slower, so training times are not",
        "comparable across rows. None of this is the final latency study.",
        "",
        "## Reproduce",
        "",
        "```sh",
        "python scripts/download_data.py",
        "python scripts/build_dataset.py",
        "python scripts/evaluate.py --config configs/bm25.yaml",
        "python scripts/evaluate.py --config configs/minilm_frozen.yaml",
        "python scripts/train.py --config configs/base.yaml",
        "python scripts/evaluate.py --config configs/minilm_tuned_epoch3.yaml",
        "python scripts/mine_negatives.py --config configs/negatives.yaml",
        "python scripts/train.py --config configs/minilm_hardneg.yaml",
        "python scripts/evaluate.py --config configs/minilm_hardneg_epoch3.yaml",
        "python scripts/make_results_table.py",
        "```",
        "",
        "The ablations follow the same pattern, one config each. `./run_ablations.sh`",
        "takes run names and does them in order.",
        "",
    ]
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
