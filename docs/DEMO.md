# Search demo

The demo serves the same BM25 plus tuned MiniLM hybrid measured in RESULTS.md.
It reads the corpus and a precomputed float16 embedding matrix at startup. Only
the query passes through the encoder during a search.

## Run with local artifacts

Build the data and model artifacts first. The default paths are:

```text
checkpoints/minilm_tuned/epoch-3/
data/processed/corpus.parquet
artifacts/answer_ids.npy
artifacts/embeddings_fp16.npy
artifacts/manifest.json
```

Start the server:

```sh
python -m scripts.start_demo
```

Then check the loaded pipeline and run a query:

```sh
curl http://localhost:7860/health
curl --get http://localhost:7860/search \
  --data-urlencode "q=How do I calculate implied volatility?" \
  --data "k=3"
```

The health response reports readiness, `bm25_dense_rrf`, and 26,152 documents.

## Remote asset layout

The deployment downloader expects one Hub model repository with transformer
files at the root and retrieval files in `demo/`:

```text
config.json
model.safetensors
tokenizer.json
tokenizer_config.json
demo/answer_ids.npy
demo/corpus.parquet
demo/embeddings_fp16.npy
demo/manifest.json
checksums.json
```

Run against a published snapshot:

```sh
python -m scripts.start_demo \
  --asset-repo OWNER/REPOSITORY \
  --asset-revision COMMIT_SHA
```

Pinning a commit instead of `main` keeps the model, corpus, and embedding matrix
on one known version.

Prepare and publish that repository with:

```sh
python scripts/prepare_demo_snapshot.py
python scripts/publish_demo_snapshot.py OWNER/REPOSITORY --public
```

The publication command reads `~/.hf_token`, checks every snapshot checksum,
and uploads the folder as one model repository commit. It defaults to private
unless `--public` is passed.

## Container

Build the image:

```sh
docker build -t quant-retrieval .
```

Run it with a published asset repository:

```sh
docker run --rm -p 7860:7860 \
  -e ASSET_REPO=OWNER/REPOSITORY \
  -e ASSET_REVISION=COMMIT_SHA \
  quant-retrieval
```

The container uses port 7860, runs as a nonroot user, and checks `/health` after
a startup grace period. The image has not been built on the development Mac
because Docker is not installed there. Build and query it before publishing a
hosted URL.

## Path overrides

The service accepts `MODEL_PATH`, `CORPUS_PATH`, `MANIFEST_PATH`,
`DOCUMENT_IDS_PATH`, and `EMBEDDINGS_PATH`. `DEVICE`, `RETRIEVAL_DEPTH`, and
`RRF_K` control runtime and fusion settings. Explicit path settings take
priority over paths inferred from a downloaded snapshot.
