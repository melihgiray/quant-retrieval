# Results

Every number here comes from the committed JSON files in `results/`, and this file is
generated from them by `scripts/make_results_table.py` rather than written by hand. Each
run ranks all 26,152 answers for every query.

The test split has not been run. Everything below is validation, 753 questions.

## Validation results

| Retriever | nDCG@10 | MRR@10 | Recall@10 | Recall@100 | Graded nDCG@10 | Index time (s) | Search p50 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.4085 | 0.3692 | 0.5339 | 0.7384 | 0.4060 | 1.39 | 5.64 |
| Frozen MiniLM | 0.4962 | 0.4518 | 0.6375 | 0.8539 | 0.5045 | 41.57 | 4.85 |
| Tuned, batch 16 | 0.5171 | 0.4714 | 0.6600 | 0.8818 | 0.5227 | 43.12 | 4.86 |
| Tuned, batch 32 | 0.5323 | 0.4889 | 0.6693 | 0.8951 | 0.5383 | 42.46 | 4.91 |
| Tuned, batch 64, epoch 1 | 0.5210 | 0.4760 | 0.6627 | 0.8884 | 0.5280 | 41.01 | 4.83 |
| Tuned, batch 64, epoch 2 | 0.5343 | 0.4899 | 0.6746 | 0.8977 | 0.5411 | 41.03 | 4.84 |
| Tuned, batch 64, epoch 3 | 0.5358 | 0.4915 | 0.6760 | 0.8924 | 0.5427 | 41.37 | 4.89 |
| Tuned, batch 128 | 0.5284 | 0.4835 | 0.6707 | 0.8938 | 0.5356 | 43.67 | 4.95 |
| Hard negatives, epoch 1 | 0.5364 | 0.4927 | 0.6746 | 0.8765 | 0.5444 | 42.82 | 4.85 |
| Hard negatives, epoch 2 | 0.5344 | 0.4898 | 0.6760 | 0.8765 | 0.5431 | 43.30 | 4.89 |
| Hard negatives, epoch 3 | 0.5367 | 0.4902 | 0.6839 | 0.8738 | 0.5451 | 42.76 | 4.85 |
| Tuned, CLS pooling | 0.4621 | 0.4188 | 0.5989 | 0.8420 | 0.4682 | 42.55 | 4.74 |
| Hybrid (BM25 + tuned) | 0.5550 | 0.5072 | 0.7065 | 0.9097 | 0.5560 | 88.98 | 60.85 |
| BM25 + reranker (undertrained) | 0.2519 | 0.2012 | 0.4170 | 0.7384 | 0.2551 | 1.35 | 281.66 |
| Tuned + reranker (undertrained) | 0.1696 | 0.1297 | 0.3015 | 0.8924 | 0.1742 | 84.10 | 481.08 |

Hybrid (BM25 + tuned) is the strongest pipeline here at 0.5550 nDCG@10, against 0.4962
for the same encoder untrained. That is +0.0587 absolute and +11.8% relative. For scale,
moving from BM25 to that untrained encoder was worth +0.0878, so domain training bought
rather less than switching to embeddings did in the first place.

Strict metrics count only the accepted answer. Graded nDCG also gives partial
credit to other nonnegative answers written for the same question.

## What the ablations say

### Hard negatives did not help

The obvious next lever after in-batch negatives is to mine wrong answers that a
retriever already ranks highly, so the model has to work for its wins. Four per question
were mined, half from BM25 and half from the tuned encoder, with every answer to the
same question excluded.

It bought +0.0008 nDCG@10, which is nothing, and cost -0.0186 Recall@100.

It did converge faster. After one epoch, mining was at 0.5364 against 0.5210 without,
and that gap had closed by epoch 3. The mined negatives carry real signal early and then
stop mattering.

The recall drop is the interesting part, and the likeliest explanation is false
negatives. On a site this narrow, the answers a retriever ranks just below the right one
are often genuinely useful answers written for a different question about the same
thing. Training the model to push those away teaches it to separate documents that
belong near each other, which is what a falling Recall@100 looks like. Excluding same-
question answers, which this already does, does not catch it.

Worth trying if this gets picked up again: skip candidates the current model already
scores very close to the positive, rather than only skipping the top hit by rank.

### Batch size matters, up to a point

In-batch negatives mean batch size is not only a speed setting. A batch of 64 asks
whether the right answer beats 63 others, a batch of 16 asks whether it beats 15.

On nDCG@10, batch 16 gives 0.5171, 32 gives 0.5323, 64 gives 0.5358, 128 gives 0.5284.
Best is 64, worth +0.0187 over batch 16, and it falls off again after that.

One caveat that matters: the learning rate was held at 2e-5 for all four. Bigger batches
therefore take proportionally fewer optimiser steps over the same three epochs, so this
measures batch size and update count together rather than batch size on its own.
Separating them means scaling the learning rate with the batch and rerunning, which is
the next experiment, not something to assert here.

### Pooling is not a free choice

Taking the first token instead of averaging them reaches 0.4621 against 0.5358, a loss
of -0.0738.

It also lands below the untrained encoder at 0.4962, which is the part worth noticing.
Three epochs of domain training did not recover what the wrong pooling gave away.

This is the expected direction rather than a surprise. all-MiniLM-L6-v2 was distilled
with mean pooling, so its first token was never trained to stand for the sequence. The
ablation is here because it is cheap and because the claim is better shown than
asserted.

### Fusing the two retrievers beats either one

BM25 and the tuned encoder fail differently. BM25 finds the exact ticker, function name
or formula that the encoder has smoothed into a general sense of the topic. The encoder
finds the answer that never repeats the question's words. Reciprocal rank fusion merges
them on rank rather than score, because BM25 sums unbounded term weights while cosine
lives in [-1, 1], and combining those numbers directly means inventing a scale factor
and then tuning it.

Hybrid reaches 0.5550 nDCG@10 against 0.5358 for the tuned encoder alone and 0.4085 for
BM25, so it is +0.0192 over the better of its two parts. Recall@100 goes from 0.8924 to
0.9097.

The recall number is the one that matters most for what comes next. A reranking stage
can only reorder what it is given, so the candidate list is a hard ceiling, and fusion
raises that ceiling before anything expensive runs.

### The reranker is not finished, and the numbers show it

A cross-encoder reads the question and one answer as a single sequence, so attention
runs across both. That is strictly more informative than comparing two vectors, and
strictly too slow to search with, so it runs last over the top 50 candidates.

It is implemented, tested, and trained for one epoch. The planned second epoch did not
complete: the machine ran out of memory partway through, and a resumed run was reduced
to about eight steps per minute against 133 in the first epoch, so it was stopped rather
than left thrashing.

BM25 + reranker (undertrained): 0.2519 nDCG@10 against 0.4085 for the same retriever
without it, -0.1566. Recall@100 is unchanged at 0.7384.

Tuned + reranker (undertrained): 0.1696 nDCG@10 against 0.5358 for the same retriever
without it, -0.3662. Recall@100 is unchanged at 0.8924.

So a half-trained reranker is worse than none, and it does more damage the better the
retriever underneath it, which is what you would expect: there is more good ordering to
destroy. Scored directly against four random documents it picks the right answer 42
percent of the time, against 20 percent for guessing, so it has learned something real
and nowhere near enough.

Recall@100 holding exactly steady is worth noting on its own. It confirms the stage only
reorders its shortlist and never drops what sits beyond it, which is the one thing a
reranker must not get wrong.

Hybrid plus reranker was not run. With both single-retriever pipelines this far down, a
third would cost ten minutes to confirm what the first two already say.

## What each row is

BM25 parameters are k1 1.2 and b 0.75. Frozen MiniLM is
`sentence-transformers/all-MiniLM-L6-v2` with no domain training. It mean pools
nonpadding tokens, normalizes each 384 dimensional embedding, and ranks by cosine
similarity.

Every tuned row is that encoder fine-tuned on the 9,924 training pairs with an
in-batch contrastive loss: each question is pulled toward its own answer and
pushed away from the other answers in its batch. Three epochs, AdamW at 2e-5 with
linear warmup over the first tenth of steps and linear decay after, gradient
clipping at 1.0, temperature 0.05, seed 17. Unless a row says otherwise it is
batch 64 and mean pooling.

All three epochs are listed for the two batch 64 runs because the checkpoint was
chosen on validation, and showing the choice is more useful than asserting it.

Timing was measured on an Apple M5 Pro. Index time covers tokenizing and embedding
the corpus, search latency covers query encoding and exact scoring against all of
it. The two largest runs used gradient checkpointing to fit in memory, which is
mathematically identical and about 30 percent slower, so training times are not
comparable across rows. None of this is the final latency study.

## Reproduce

```sh
python scripts/download_data.py
python scripts/build_dataset.py
python scripts/evaluate.py --config configs/bm25.yaml
python scripts/evaluate.py --config configs/minilm_frozen.yaml
python scripts/train.py --config configs/base.yaml
python scripts/evaluate.py --config configs/minilm_tuned_epoch3.yaml
python scripts/mine_negatives.py --config configs/negatives.yaml
python scripts/train.py --config configs/minilm_hardneg.yaml
python scripts/evaluate.py --config configs/minilm_hardneg_epoch3.yaml
python scripts/make_results_table.py
```

The ablations follow the same pattern, one config each. `./run_ablations.sh`
takes run names and does them in order.
