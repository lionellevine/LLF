# Extending LLF

This guide is for Ananya and other students adding data, adapters, score definitions, or filtering experiments without changing the provenance of the released `v0.1.0` result.

## Preserve the existing release

Treat `data/logprobs/*.parquet`, `data/bundle_manifest.json`, `data/reference/`, and `export_manifest.json` as immutable. New work belongs in a versioned directory or branch. Never overwrite an old summary with a new model revision, adapter stage, formula, target matrix, or seed.

For every extension, record:

- source dataset repository, immutable revision, split/file, row count, and license terms;
- base model and adapter repository revisions plus adapter weight hashes;
- tokenizer/chat-template and inference runtime versions;
- score formula, aggregation, calibration, random seed, and resampling counts;
- output SHA-256 values and whether source text is present.

Run `oct-llf validate-data ... --scan-repository .` before publishing text-free data.

## Add a preference dataset

1. Obtain the source under its own terms; do not add raw prompts or responses to this repository.
2. Run `scripts/score_oct_pairs.py` against a pinned model and, when applicable, a pinned adapter revision.
3. Convert the text-free output to the 61-column public schema in `docs/SCHEMA.md` when scoring all ten adapters.
4. Store it under a new versioned bundle and create a new manifest entry with rows, bytes, SHA-256, and `contains_raw_text: false`.
5. Validate formulas from raw totals; never trust only stored derived columns.

## Add an adapter or checkpoint stage

Adding an adapter changes a matrix axis and therefore creates a new bundle version. Record whether the checkpoint is combined DPO + introspective SFT, pure DPO-final, pure SFT, or another stage. Do not append a pure-DPO column to the current combined-adapter matrix and call it the same experiment.

A full pure-DPO comparison should pin ten DPO-final checkpoints, score the same ten datasets, and report its own 10×10 matrix, behavior target, bootstrap, and block-permutation outputs.

## Add a score definition

Implement the formula first as a pure scalar function and a vectorized array function. Add tests showing equivalence on fixed inputs and invalid-length handling. Give the formula a new name; do not redefine `sum`, `lls`, or `separate`.

Re-run the complete sensitivity grid if the new score is compared with existing results. Pre-register which formula is primary before observing the target association.

## Run a filtering experiment

Filtering is a downstream intervention, not evidence that the score is causal. Save the selected row hashes and selection rule as a separate manifest. Compare against matched-size random and length-matched controls, keep training recipes fixed, and evaluate on held-out behavioral data. Report both intended effects and cross-trait side effects.

## Before opening a pull request

- run Ruff, mypy, the full test suite, both package builds, bundle validation, reference reproduction, and the repository public-only scan;
- confirm no file is 100 MB or larger and no model/checkpoint binary is present;
- inspect `git status` and stage only intended public files;
- update `docs/PROVENANCE.md`, `docs/VALIDATION.md`, and citation metadata when identities or claims change.
