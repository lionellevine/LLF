# Repository data

`logprobs/` contains ten immutable, text-free Parquet files. `bundle_manifest.json` records each file's trait, row count, byte size, SHA-256, and exact formula round-trip result. Run:

```bash
oct-llf validate-data data --manifest data/bundle_manifest.json
```

`reference/` contains the fixed primary calibrated matrix, the pinned `oct-v2` behavior target, all 18 expected matrix hashes, and the analysis specification that defines orientation and sensitivity choices. Run `python scripts/reproduce_results.py` to rebuild those matrices and fail on any headline or hash mismatch. The Parquet files are repository data, not Python package payloads, and are excluded from wheel and source distributions.

See the schema, provenance, validation history, and data card under `docs/`. The data is not relicensed under MIT; source non-commercial/research and stricter-source terms remain applicable.
