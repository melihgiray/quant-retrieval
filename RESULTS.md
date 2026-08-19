# Results

Every number here comes from the committed JSON files in `results/`, and this file is
generated from them by `scripts/make_results_table.py` rather than written by hand. Each
run ranks all 26,152 answers for every query.

The test split has not been run. Everything below is validation, 753 questions.

## Validation results

| Retriever | nDCG@10 | MRR@10 | Recall@10 | Recall@100 | Graded nDCG@10 | Index time (s) | Search p50 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.4085 | 0.3692 | 0.5339 | 0.7384 | 0.4060 | 1.38 | 5.67 |
| Frozen MiniLM | 0.4962 | 0.4518 | 0.6375 | 0.8539 | 0.5045 | 40.44 | 4.73 |
| Tuned, batch 16 | 0.5171 | 0.4714 | 0.6600 | 0.8818 | 0.5227 | 39.90 | 4.71 |
| Tuned, batch 32 | 0.5323 | 0.4889 | 0.6693 | 0.8951 | 0.5383 | 39.57 | 4.77 |
| Tuned, batch 64, epoch 1 | 0.5210 | 0.4760 | 0.6627 | 0.8884 | 0.5280 | 39.95 | 5.09 |
| Tuned, batch 64, epoch 2 | 0.5343 | 0.4899 | 0.6746 | 0.8977 | 0.5411 | 39.71 | 4.78 |
| Tuned, batch 64, epoch 3 | 0.5358 | 0.4915 | 0.6760 | 0.8924 | 0.5427 | 40.33 | 4.73 |
| Tuned, batch 128 | 0.5284 | 0.4835 | 0.6707 | 0.8938 | 0.5356 | 40.44 | 4.86 |
| Hard negatives, epoch 1 | 0.5364 | 0.4927 | 0.6746 | 0.8765 | 0.5444 | 40.29 | 4.76 |
| Hard negatives, epoch 2 | 0.5344 | 0.4898 | 0.6760 | 0.8765 | 0.5431 | 40.00 | 4.92 |
| Hard negatives, epoch 3 | 0.5367 | 0.4902 | 0.6839 | 0.8738 | 0.5451 | 39.77 | 4.93 |
| Tuned, CLS pooling | 0.4621 | 0.4188 | 0.5989 | 0.8420 | 0.4682 | 39.82 | 4.73 |
| Hybrid (BM25 + tuned) | 0.5550 | 0.5072 | 0.7065 | 0.9097 | 0.5560 | 41.06 | 12.81 |
| BM25 + reranker (mined only) | 0.2695 | 0.2150 | 0.4462 | 0.7384 | 0.2719 | 1.42 | 112.92 |
| Tuned + reranker (mined only) | 0.1932 | 0.1467 | 0.3453 | 0.8924 | 0.1981 | 40.79 | 113.45 |
| Hybrid + reranker (mined only) | 0.2351 | 0.1790 | 0.4197 | 0.9097 | 0.2405 | 43.33 | 120.50 |
| BM25 + reranker (mixed negatives) | 0.3599 | 0.3043 | 0.5392 | 0.7384 | 0.3607 | 1.40 | 110.70 |
| Tuned + reranker (mixed negatives) | 0.2747 | 0.2204 | 0.4542 | 0.8924 | 0.2791 | 39.88 | 108.50 |
| Hybrid + reranker (mixed negatives) | 0.3076 | 0.2457 | 0.5113 | 0.9097 | 0.3118 | 40.94 | 116.30 |

Hybrid (BM25 + tuned) has the highest nDCG@10 here at 0.5550, against 0.4962 for the
same encoder untrained. Fine-tuning alone accounts for most of that: it is worth +0.0396
([+0.0251, +0.0546], p = 0.000). For scale, moving from BM25 to that untrained encoder
was worth +0.0878 ([+0.0543, +0.1210], p = 0.000), so domain training bought rather less
than switching to embeddings did in the first place.

Every comparison below carries a 95 percent confidence interval and a p value from a
paired bootstrap: resample the 753 questions ten thousand times, recompute both systems
on each resample, and see whether the difference between them keeps its sign. Paired,
because both systems answered the same questions, so the large variance from some
questions simply being harder cancels instead of drowning the effect. Two of the
differences this file used to describe as real do not survive that test, and they are
marked where they appear.

Strict metrics count only the accepted answer. Graded nDCG also gives partial
credit to other nonnegative answers written for the same question.

## What the ablations say

### Hard negatives did not help

The obvious next lever after in-batch negatives is to mine wrong answers that a
retriever already ranks highly, so the model has to work for its wins. Four per question
were mined, half from BM25 and half from the tuned encoder, with every answer to the
same question excluded.

It bought +0.0008 nDCG@10 ([-0.0135, +0.0153], p = 0.91, which does not exclude zero, so
this difference is not distinguishable from noise), and cost -0.0186 Recall@100
([-0.0332, -0.0040], p = 0.014). So the ranking gain really is nothing, and the recall
loss really is something. Reporting one without the other would have been the flattering
half.

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
The climb from 16 to 64 is +0.0187 and it falls off again after that.

The peak is softer than it looks. Batch 32 against batch 64 is +0.0035 ([-0.0034,
+0.0104], p = 0.32, which does not exclude zero, so this difference is not
distinguishable from noise). So the honest reading is that batch size matters over the
range 16 to 64 and that 32 and 64 are indistinguishable on this validation set. An
earlier version of this file said the peak was at 64, which was reading a ranking off
differences the data does not support.

One caveat that matters: the learning rate was held at 2e-5 for all four. Bigger batches
therefore take proportionally fewer optimiser steps over the same three epochs, so this
measures batch size and update count together rather than batch size on its own.
Separating them means scaling the learning rate with the batch and rerunning, which is
the next experiment, not something to assert here.

### Pooling is not a free choice

Taking the first token instead of averaging them reaches 0.4621 against 0.5358, a loss
of -0.0738 ([-0.0912, -0.0568], p = 0.000).

It also lands below the untrained encoder at 0.4962, which is the part worth noticing.
Three epochs of domain training did not recover what the wrong pooling gave away.

This is the expected direction rather than a surprise. all-MiniLM-L6-v2 was distilled
with mean pooling, so its first token was never trained to stand for the sequence. The
ablation is here because it is cheap and because the claim is better shown than
asserted.

### Fusing the two retrievers buys recall, not ranking

BM25 and the tuned encoder fail differently. BM25 finds the exact ticker, function name
or formula that the encoder has smoothed into a general sense of the topic. The encoder
finds the answer that never repeats the question's words. Reciprocal rank fusion merges
them on rank rather than score, because BM25 sums unbounded term weights while cosine
lives in [-1, 1], and combining those numbers directly means inventing a scale factor
and then tuning it.

Hybrid reaches 0.5550 nDCG@10 against 0.5358 for the tuned encoder alone, which is
+0.0192 ([-0.0070, +0.0453], p = 0.15, which does not exclude zero, so this difference
is not distinguishable from noise).

Recall@100 is the different story. It goes from 0.8924 to 0.9097, +0.0173 ([+0.0013,
+0.0345], p = 0.040), which does hold up.

So fusion earns its place by finding answers the encoder alone misses, not by ordering
them better. That is a narrower claim than the one this file made before the bootstrap
was run, and it is the one the data supports. It also happens to be the more useful
half: a reranking stage can reorder a candidate list but cannot conjure a document that
is not in it, so the ceiling fusion raises is the ceiling that matters.

Worth saying plainly: with 753 questions, a difference of about 0.02 in nDCG@10 sits
right at the edge of what this evaluation set can resolve. Reaching a verdict on
fusion's ranking effect needs more questions, not more argument.

### The reranker: a diagnosis that was right and a fix that was not enough

A cross-encoder reads the question and one answer as a single sequence, so attention
runs across both. Strictly more informative than comparing two vectors, strictly too
slow to search with, so it runs last over the top 50 candidates.

The first one made every pipeline worse, and undertraining was not the reason. Its
training accuracy climbed, its scoring head ended almost orthogonal to its
initialisation, its backbone moved a normal amount, and yet against four documents
picked completely at random it scored 43 percent on validation questions and 28 percent
on questions it had trained on, where chance is 20. Nothing merely undertrained fails on
its own training data against off-topic distractors.

The reading was that it had been trained on the wrong distribution. Every negative it
ever saw was a mined hard negative, a plausible answer to a similar question, so it
learned to make fine distinctions inside a narrow band and never learned the coarse one.
At search time most of its 50 candidates are exactly the coarse case.

That diagnosis was testable, so it was tested: retrain with two mined negatives and two
drawn uniformly from the corpus, everything else identical. Against four random
documents the new model scores 85.0 percent on validation and 86.0 on training
questions, against 43.3 and 28.3 before. The training split is no longer the worse of
the two, which was the specific symptom the diagnosis predicted would go away.

BM25: 0.4085 with no reranker, 0.2695 with the mined-only one, 0.3599 with the mixed
one.

Tuned: 0.5358 with no reranker, 0.1932 with the mined-only one, 0.2747 with the mixed
one.

Hybrid: 0.5550 with no reranker, 0.2351 with the mined-only one, 0.3076 with the mixed
one.

So the fix moved every pipeline in the right direction and not remotely far enough. On
the strongest base the reranker still costs -0.2611 nDCG@10 ([-0.2908, -0.2315], p =
0.000).

Which is the honest shape of the result: the hypothesis was right about the cause and
the remedy was insufficient. 85 percent against four random documents is a real
improvement and still weak for a cross-encoder, and a model that hesitates on five
candidates has no chance of ordering fifty. The likeliest remaining causes are the size
of the model, six layers and 22 million parameters doing a job usually given to
something larger, and a training group of five candidates when inference presents fifty.

Recall@100 is identical with and without the reranker in all six rows. That is the stage
behaving correctly even as its scores do not: it reorders its shortlist and never drops
what lies beyond it, which is the one thing a reranking stage must not get wrong, and
there is a test for it.

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
