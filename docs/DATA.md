# The dataset

Every number here comes from `results/dataset_stats.json`, which the build
script writes. If you change the pipeline, rerun the build and the numbers
change with it.

    python scripts/download_data.py
    python scripts/build_dataset.py

## Source

The public Stack Exchange data dump for quant.stackexchange.com, from
archive.org. The archive we used is dated 6 April 2024 and is 55 MB compressed.
Stack Exchange republishes these, so a later dump will give different counts.

User contributions are licensed CC BY-SA 4.0 by their authors. The dump is not
redistributed in this repository. `scripts/download_data.py` fetches it.

## The task

Answer retrieval. A query is a question (title first, then body). The corpus is
every answer on the site. Given a question nobody has answered yet, find the
answers that address it.

The corpus is answers alone, not questions glued to their answers. Pairing them
would let a lexical baseline match the query against a copy of itself sitting
inside the document, which makes the task look easy for the wrong reason.

## Tables

Three parquet files in `data/processed/`, the layout most IR tooling expects.

`corpus.parquet`, 26,152 documents. Columns: `answer_id`, `question_id`,
`score`, `text`. Median document is 679 characters, 109 words. The 90th
percentile is 2,058 characters, which matters because the encoder truncates
at 256 tokens, so roughly the top quarter of documents get cut. That is a real
limitation and a candidate for a passage splitting ablation later.

`queries.parquet`, 11,474 queries. Columns: `question_id`, `creation_date`,
`title`, `tags`, `text`, `split`, `label_source`.

`qrels.parquet`, 18,959 judgements. Columns: `question_id`, `answer_id`,
`grade`, `label_source`. That is 1.65 judgements per query.

## How answers are judged

Grade 2, the primary judgement, 11,474 of them. The accepted answer where there
is one. Where the asker never accepted anything, the top voted answer stands in,
but only if it scored at least 2 and beat the runner up outright. A tie means
nobody can say which answer the asker wanted, so the question is dropped.

Grade 1, 7,485 of them. Any other answer on a judged question that was not voted
below zero. Someone searching the site wants an answer to their question, not
specifically the one with the green tick, and this also stops the metric from
punishing a model that puts the second best answer first. The evaluation harness
can ignore grade 1 and score accepted-only retrieval instead.

Only 7,947 of the 22,177 questions have an accepted answer, about 36 percent.
The top voted fallback is what takes the usable set from roughly 8k to 11.5k.

Nothing is filtered out of the corpus. Short answers, unhelpful answers and
downvoted answers all stay in and can be retrieved. Removing them would tune the
benchmark instead of the model.

## Splits

By date, not at random. Questions are sorted by creation date, the newest 10
percent become test, the 10 percent before that become validation, the rest is
training.

| split | queries | first question | last question |
| --- | --- | --- | --- |
| train | 9,924 | 2010-09-19 | 2020-10-21 |
| val | 753 | 2020-10-21 | 2021-12-13 |
| test | 797 | 2021-12-13 | 2024-03-31 |

A random split would let the model train on a question asked the same week as a
test question, often about the same paper by the same handful of people, and
every metric would come out flattering.

Time alone does not close the leak. The same question gets asked again years
later. Duplicate links from `PostLinks.xml` and exactly matching normalized
titles are unioned into groups, and any group spanning two splits keeps only its
members in the strongest split (test beats validation beats train). That removed
21 questions.

Validation and test carry accepted answers only. The top voted fallback is good
enough to learn from and too noisy to report on, so 935 held out questions
without an accepted answer were dropped. Training keeps both: 7,934 accepted and
3,540 top voted.

## Known limitations

The 256 token truncation cuts the longest quarter of documents.

The top voted fallback in training is a heuristic. Some of those answers are not
what the asker wanted.

Grade 1 siblings make the metric more forgiving than strict accepted-only
retrieval. Both are reported.

The test window (December 2021 to March 2024) is a different market period from
most of training, which is part of the point and also a confound. A model can
lose on test because the vocabulary moved, not because it retrieves worse.
