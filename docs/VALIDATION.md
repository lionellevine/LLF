# Validation claims and their boundaries

The history contains three distinct checks. They answer different questions and must not be merged into one claim.

## 1. Token parity passed

On the fixed validation rows, response encoding matched the OpenRLHF reference exactly for input token IDs, prompt boundaries, and response-token counts. Checks also covered the Qwen default system header, one assistant-end token, and exclusion of the trailing newline. This supports token-boundary compatibility; it does not establish numerical agreement between inference engines.

## 2. Historical primitive engine gate failed

A preregistered primitive gate required the maximum absolute difference between Transformers and vLLM mean log-probability per response token to be at most **0.01**, over 16 pairs, three model states, and 96 response comparisons.

- Default attention maximum: **0.0234732471**.
- Matched-attention maximum: **0.0324486170**.

Both exceed 0.01. The primitive gate failed, full scoring was not unlocked under that original plan, and it was **not retroactively passed**. Exact token parity does not repair this numerical failure.

A subsequent derived-delta inspection on the same 16 fixed rows found cancellation of some primitive discrepancies and high rank agreement. That cancellation/rank audit was explicitly **descriptive**. It had too few rows to establish full-dataset trait-mean stability, and its base-vs-base zero was an algebraic self-check rather than an engine determinism test.

## 3. Later within-vLLM derived-delta repeatability passed

A new preregistered check tested the endpoint actually used downstream: base-relative chosen-minus-rejected derived deltas. It used 512 unique prompts and two cold starts with fresh vLLM engines, deliberately different response batch sizes (64 versus 17), and different score-shard sizes (512 versus 127).

All preregistered gates passed:

- trait-mean vector: Spearman 1.0, identical top-three trait set, and maximum repeat difference 0.0003410330 (0.00279655 of pooled IQR), below the allowed 0.0121947965;
- every per-trait rank gate passed;
- every per-trait top-decile and bottom-decile overlap gate passed.

This supports repeatability of the derived endpoint **within vLLM** across those two cold starts and batching/sharding choices. It does not show primitive Transformers-vLLM equivalence and does not alter the earlier failed 0.01 gate. Per-pair absolute differences were descriptive and had no 0.01 acceptance threshold in this later plan.

## Bundle integrity

`oct-llf validate-data data` independently checks all ten file SHA-256 values, sizes, row counts, exact shared schema, positive lengths, finite values, absence of raw-text columns/metadata, and exact formula round-trip. The expected totals are 10 files, 84,803 rows, 47,966,184 bytes, and maximum formula error 0.

## Headline result replay

`python scripts/reproduce_results.py` rebuilds all 18 raw/calibrated score matrices, checks their recorded SHA-256 values, verifies the pinned `oct-v2` behavioral target and orientation sentinels, and fails unless it reproduces:

- combined off-diagonal Pearson `0.104758806591627` over 90 cells;
- humor-column Pearson `0.8069556769263959`;
- single-humor separate-response Spearman `-0.38333333333333336`;
- single-humor unnormalized Pearson `-0.6215649381632113` within `1e-6`;
- the original 1,000-bootstrap interval and 100,000-permutation p-values in full mode.

The score and behavior matrices share `C,T` axes directly: adapter trait `C` corresponds to evaluation trait `C`, and preference dataset `T` corresponds to training trait `T`. A transpose is a failing orientation error, not a sensitivity choice.
