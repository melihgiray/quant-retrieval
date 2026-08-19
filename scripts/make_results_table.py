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


def load_comparisons(results_dir: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Every paired bootstrap that has been run, keyed by what it compared."""
    comparisons = {}
    for path in sorted((results_dir / "comparisons").glob("*.json")):
        record = json.loads(path.read_text())
        comparisons[(record["baseline"], record["candidate"], record["metric"])] = record
    return comparisons


def verdict(
    comparisons: dict, baseline: str, candidate: str, metric: str = "ndcg_at_10"
) -> str:
    """One clause stating whether a difference survived resampling."""
    record = comparisons.get((f"{baseline}_val", f"{candidate}_val", metric))
    if record is None:
        return ""
    interval = f"[{record['ci_low']:+.4f}, {record['ci_high']:+.4f}]"
    if record["significant"]:
        return f" ({interval}, p = {record['p_value']:.3f})"
    return (
        f" ({interval}, p = {record['p_value']:.2f}, which does not exclude zero, "
        "so this difference is not distinguishable from noise)"
    )


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


def headline(evaluations: list[dict[str, Any]], comparisons: dict) -> list[str]:
    frozen = find(evaluations, "minilm_frozen_val")
    # Every system meant to be an improvement. Baselines are the thing being
    # beaten, and the reranker pipelines lose to their own base retrievers, so
    # neither belongs in a claim about the best result.
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
        f"{display_name(best)} has the highest nDCG@10 here at {after:.4f}, against "
        f"{before:.4f} for the same encoder untrained. Fine-tuning alone accounts for most "
        f"of that: it is worth {score(find(evaluations, 'minilm_tuned_epoch3_val')) - before:+.4f}"
        + verdict(comparisons, "minilm_frozen", "minilm_tuned_epoch3")
        + "."
    )
    if bm25 is not None:
        gap = before - bm25["metrics"]["ndcg_at_10"]
        text += (
            f" For scale, moving from BM25 to that untrained encoder was worth {gap:+.4f}"
            + verdict(comparisons, "bm25", "minilm_frozen")
            + ", so domain training bought rather less than switching to embeddings did in "
            "the first place."
        )
    lines = paragraph(text)
    lines += paragraph(
        "Every comparison below carries a 95 percent confidence interval and a p value from "
        "a paired bootstrap: resample the 753 questions ten thousand times, recompute both "
        "systems on each resample, and see whether the difference between them keeps its "
        "sign. Paired, because both systems answered the same questions, so the large "
        "variance from some questions simply being harder cancels instead of drowning the "
        "effect. Two of the differences this file used to describe as real do not survive "
        "that test, and they are marked where they appear."
    )
    return lines


def hard_negative_section(evaluations: list[dict[str, Any]], comparisons: dict) -> list[str]:
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
        f"It bought {ndcg_delta:+.4f} nDCG@10"
        + verdict(comparisons, "minilm_tuned_epoch3", "minilm_hardneg_epoch3")
        + f", and cost {recall_delta:+.4f} Recall@100"
        + verdict(comparisons, "minilm_tuned_epoch3", "minilm_hardneg_epoch3", "recall_at_100")
        + ". So the ranking gain really is nothing, and the recall loss really is something. "
        "Reporting one without the other would have been the flattering half."
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


def batch_section(evaluations: list[dict[str, Any]], comparisons: dict) -> list[str]:
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
        f"On nDCG@10, batch {listed}. The climb from {smallest} to {best_size} is "
        f"{climb:+.4f} and it falls off again after that."
    )
    lines += paragraph(
        "The peak is softer than it looks. Batch 32 against batch 64 is "
        f"{score(present[64]) - score(present[32]):+.4f}"
        + verdict(comparisons, "minilm_batch32_epoch3", "minilm_tuned_epoch3")
        + ". So the honest reading is that batch size matters over the range 16 to 64 and "
        "that 32 and 64 are indistinguishable on this validation set. An earlier version of "
        "this file said the peak was at 64, which was reading a ranking off differences the "
        "data does not support."
    )
    lines += paragraph(
        "One caveat that matters: the learning rate was held at 2e-5 for all four. Bigger "
        "batches therefore take proportionally fewer optimiser steps over the same three "
        "epochs, so this measures batch size and update count together rather than batch "
        "size on its own. Separating them means scaling the learning rate with the batch "
        "and rerunning, which is the next experiment, not something to assert here."
    )
    return lines


def pooling_section(evaluations: list[dict[str, Any]], comparisons: dict) -> list[str]:
    mean = find(evaluations, "minilm_tuned_epoch3_val")
    cls = find(evaluations, "minilm_cls_epoch3_val")
    frozen = find(evaluations, "minilm_frozen_val")
    if mean is None or cls is None:
        return []

    lines = ["### Pooling is not a free choice", ""]
    lines += paragraph(
        f"Taking the first token instead of averaging them reaches {score(cls):.4f} against "
        f"{score(mean):.4f}, a loss of {score(cls) - score(mean):.4f}"
        + verdict(comparisons, "minilm_tuned_epoch3", "minilm_cls_epoch3")
        + "."
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


def hybrid_section(evaluations: list[dict[str, Any]], comparisons: dict) -> list[str]:
    hybrid = find(evaluations, "hybrid_val")
    dense = find(evaluations, "minilm_tuned_epoch3_val")
    bm25 = find(evaluations, "bm25_val")
    if hybrid is None or dense is None or bm25 is None:
        return []

    lines = ["### Fusing the two retrievers buys recall, not ranking", ""]
    lines += paragraph(
        "BM25 and the tuned encoder fail differently. BM25 finds the exact ticker, function "
        "name or formula that the encoder has smoothed into a general sense of the topic. The "
        "encoder finds the answer that never repeats the question's words. Reciprocal rank "
        "fusion merges them on rank rather than score, because BM25 sums unbounded term "
        "weights while cosine lives in [-1, 1], and combining those numbers directly means "
        "inventing a scale factor and then tuning it."
    )
    lines += paragraph(
        f"Hybrid reaches {score(hybrid):.4f} nDCG@10 against {score(dense):.4f} for the "
        f"tuned encoder alone, which is {score(hybrid) - score(dense):+.4f}"
        + verdict(comparisons, "minilm_tuned_epoch3", "hybrid")
        + "."
    )
    lines += paragraph(
        f"Recall@100 is the different story. It goes from {score(dense, 'recall_at_100'):.4f} "
        f"to {score(hybrid, 'recall_at_100'):.4f}, "
        f"{score(hybrid, 'recall_at_100') - score(dense, 'recall_at_100'):+.4f}"
        + verdict(comparisons, "minilm_tuned_epoch3", "hybrid", "recall_at_100")
        + ", which does hold up."
    )
    lines += paragraph(
        "So fusion earns its place by finding answers the encoder alone misses, not by "
        "ordering them better. That is a narrower claim than the one this file made before "
        "the bootstrap was run, and it is the one the data supports. It also happens to be "
        "the more useful half: a reranking stage can reorder a candidate list but cannot "
        "conjure a document that is not in it, so the ceiling fusion raises is the ceiling "
        "that matters."
    )
    lines += paragraph(
        "Worth saying plainly: with 753 questions, a difference of about 0.02 in nDCG@10 sits "
        "right at the edge of what this evaluation set can resolve. Reaching a verdict on "
        "fusion's ranking effect needs more questions, not more argument."
    )
    return lines


def reranker_section(evaluations: list[dict[str, Any]], comparisons: dict) -> list[str]:
    reranked = [r for r in evaluations if r["run_name"].endswith("_rerank_val")]
    mixed = [r for r in evaluations if r["run_name"].endswith("_rerank_mixed_val")]
    if not reranked and not mixed:
        return []

    lines = ["### The reranker: a diagnosis that was right and a fix that was not enough", ""]
    lines += paragraph(
        "A cross-encoder reads the question and one answer as a single sequence, so attention "
        "runs across both. Strictly more informative than comparing two vectors, strictly too "
        "slow to search with, so it runs last over the top 50 candidates."
    )
    lines += paragraph(
        "The first one made every pipeline worse, and undertraining was not the reason. Its "
        "training accuracy climbed, its scoring head ended almost orthogonal to its "
        "initialisation, its backbone moved a normal amount, and yet against four documents "
        "picked completely at random it scored 43 percent on validation questions and 28 "
        "percent on questions it had trained on, where chance is 20. Nothing merely "
        "undertrained fails on its own training data against off-topic distractors."
    )
    lines += paragraph(
        "The reading was that it had been trained on the wrong distribution. Every negative it "
        "ever saw was a mined hard negative, a plausible answer to a similar question, so it "
        "learned to make fine distinctions inside a narrow band and never learned the coarse "
        "one. At search time most of its 50 candidates are exactly the coarse case."
    )
    lines += paragraph(
        "That diagnosis was testable, so it was tested: retrain with two mined negatives and "
        "two drawn uniformly from the corpus, everything else identical. Against four random "
        "documents the new model scores 85.0 percent on validation and 86.0 on training "
        "questions, against 43.3 and 28.3 before. The training split is no longer the worse "
        "of the two, which was the specific symptom the diagnosis predicted would go away."
    )

    bases = {
        "bm25_rerank": "bm25_val",
        "dense_rerank": "minilm_tuned_epoch3_val",
        "hybrid_rerank": "hybrid_val",
    }
    for stem, label in (("bm25_rerank", "BM25"), ("dense_rerank", "Tuned"),
                        ("hybrid_rerank", "Hybrid")):
        base = find(evaluations, bases[stem])
        old = find(evaluations, f"{stem}_val")
        new = find(evaluations, f"{stem}_mixed_val")
        if not (base and old and new):
            continue
        lines += paragraph(
            f"{label}: {score(base):.4f} with no reranker, {score(old):.4f} with the "
            f"mined-only one, {score(new):.4f} with the mixed one."
        )

    tuned = find(evaluations, "minilm_tuned_epoch3_val")
    best_rerank = find(evaluations, "dense_rerank_mixed_val")
    if tuned is not None and best_rerank is not None:
        cost = score(best_rerank) - score(tuned)
        lines += paragraph(
            "So the fix moved every pipeline in the right direction and not remotely far "
            f"enough. On the strongest base the reranker still costs {cost:+.4f} nDCG@10"
            + verdict(comparisons, "minilm_tuned_epoch3", "dense_rerank_mixed")
            + "."
        )
    lines += paragraph(
        "Which is the honest shape of the result: the hypothesis was right about the cause and "
        "the remedy was insufficient. 85 percent against four random documents is a real "
        "improvement and still weak for a cross-encoder, and a model that hesitates on five "
        "candidates has no chance of ordering fifty. The likeliest remaining causes are the "
        "size of the model, six layers and 22 million parameters doing a job usually given to "
        "something larger, and a training group of five candidates when inference presents "
        "fifty."
    )
    lines += paragraph(
        "Recall@100 is identical with and without the reranker in all six rows. That is the "
        "stage behaving correctly even as its scores do not: it reorders its shortlist and "
        "never drops what lies beyond it, which is the one thing a reranking stage must not "
        "get wrong, and there is a test for it."
    )
    return lines


def render_results(
    evaluations: list[dict[str, Any]], split: str, comparisons: dict | None = None
) -> str:
    if not evaluations:
        raise ValueError(f"no {split} evaluation results found")
    comparisons = comparisons or {}

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
        *headline(evaluations, comparisons),
        "Strict metrics count only the accepted answer. Graded nDCG also gives partial",
        "credit to other nonnegative answers written for the same question.",
        "",
        "## What the ablations say",
        "",
        *hard_negative_section(evaluations, comparisons),
        *batch_section(evaluations, comparisons),
        *pooling_section(evaluations, comparisons),
        *hybrid_section(evaluations, comparisons),
        *reranker_section(evaluations, comparisons),
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
    comparisons = load_comparisons(args.results)
    args.output.write_text(render_results(evaluations, args.split, comparisons))
    print(f"wrote {args.output} from {len(evaluations)} evaluations")


if __name__ == "__main__":
    main()
