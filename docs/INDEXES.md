# Verified index exports and precision evaluation

An export binds ordered answer IDs to document vectors. Shapes alone cannot
prove those vectors still describe the current answer text. New exports record
the corpus fingerprint, payload checksums, seed, pooling and encoding settings.

## Create a new version

```sh
python -m scripts.export_index --checkpoint checkpoints/minilm_tuned/epoch-3 \
  --out artifacts/quant-v2 --batch-size 128 --max-length 256 --seed 17
```

This command encodes the full corpus. Run it on a suitable machine when ready
for that work. It refuses an existing output directory before loading the model.
Use a fresh version name rather than overwriting an index in use.

IDs must be unique positive int64 values and text must be nonempty. Encoder
output must have the expected number of finite, normalized rows. Both stored
precisions are checked. All payloads and the manifest are written in a sibling
staging directory, then published together. Failure removes the staging data,
not earlier exports. The output contains:

- `answer_ids.npy`, in the original corpus order.
- `embeddings_fp32.npy` and `embeddings_fp16.npy`, with identical row alignment.
- `manifest.json`, including schema version, dimensions, corpus fingerprint and
  SHA-256 hashes for all three arrays.

Checksums detect changed files relative to the manifest. They are not signatures
and do not authenticate a downloaded model or its publisher. Checkpoint identity
is still a recorded path, not a hash of its weights. Preserve the checkpoint
files used for the export; replacing weights at the same path is not detected.

## Inspect storage without running the model

```sh
python -m scripts.inspect_index --artifacts artifacts/quant-v2 \
  --output results/quant_v2_storage.json
```

The inspector reads arrays with memory mapping and checks norms in row blocks.
It reports dtype, array and file size, maximum absolute float16 conversion error,
root mean square error and maximum unit-norm error. It refuses an existing
report path. No encoder is loaded and no relevance metrics are calculated.

Legacy exports without checksums remain inspectable, but the report marks
`payload_checksums_verified` false. Do not add current hashes to an old manifest
and claim that verifies its original production. New versioned manifests must
carry their full checksum map.

## Compare retrieval quality at both precisions

```sh
python -m scripts.evaluate --config configs/index_fp32.yaml --save-rankings
python -m scripts.evaluate --config configs/index_fp16.yaml --save-rankings
python -m scripts.compare_runs --baseline results/index_fp32_val.json \
  --candidate results/index_fp16_val.json --metric ndcg_at_10
```

Both configs expect `artifacts/quant-v2`. Change both paths together if the export
has another name. The `precomputed_dense` retriever requires a checksum-verified
export and an exact match of corpus IDs, text and ordering. Legacy exports need
to be regenerated for this evaluation mode. The retrieval harness still ranks
the entire corpus and applies the usual validation judgments.

Only query text is encoded. Model path, pooling and maximum sequence length
come from the export contract. Stored float16 vectors do not imply float16 query
encoding or float16 model inference. Index timing measures validation and loading,
not document encoding, so it is not directly comparable with a freshly encoded
index's build time.

The two configs are experiment definitions, not evidence that precision has no
quality cost. Run and retain their results before making that claim. A small
numerical error can still change rankings near a tie. Dense selection now uses
document ID to resolve every boundary tie, including ties crossing the cutoff.
Compare reruns at the same source revision when auditing historical results.

## Consumers

ANN sweeps verify new payload checksums before indexing and record whether
inputs could be verified. Serving checks new corpus fingerprints and selected
array checksums before loading the query encoder. Legacy serving manifests keep
their previous structural checks. An index already loaded by a process must not
be edited in place; publish a new directory and restart the consumer instead.
