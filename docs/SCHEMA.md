# Public Parquet schema

All ten files have an identical schema and one file per preference-dataset trait.

## Identity and provenance columns

- `dataset_trait` (string): dataset/file trait.
- `source_revision` (string): immutable source data revision.
- `source_row_id` (int64): row identifier in that source revision.
- `prompt_hash`, `chosen_hash`, `rejected_hash` (string): lowercase SHA-256 of UTF-8 text. These hashes can be linkable; they are not raw text and not an anonymity guarantee.

## Token and likelihood columns

- `prompt_tokens`, `chosen_response_tokens`, `rejected_response_tokens` (int64, positive).
- `base_chosen_logprob`, `base_rejected_logprob` (float64): response-token total log-probabilities under the base model.
- `{trait}_chosen_logprob`, `{trait}_rejected_logprob` (float64): corresponding totals under each adapter.

## Derived columns

For every adapter trait:

- `delta_sum_{trait}`
- `delta_lls_{trait}`
- `delta_separate_{trait}`

The formulas are defined in the README. Every value round-trips exactly from the stored raw totals and token lengths (maximum absolute error 0 in the bundled files).

## Exclusions

There are no columns containing prompt, chosen, rejected, message, response, completion, or other raw text. Schema metadata contains a JSON provenance record and a `contains_raw_text: false` assertion. The validator rejects hidden raw-text fields and unexpected string columns.
