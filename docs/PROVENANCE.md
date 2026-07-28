# Provenance

## Immutable identities

- Source preference data: `maius/OpenCharacterTraining-data` revision `2577813a6a435d21051c0548ff2f29dc897212d7`.
- Base model: `Qwen/Qwen2.5-7B-Instruct` revision `a09a35458c702b33eeacc393d103063234e8bc28`.
- Combined public adapter repository: `maius/qwen-2.5-7b-it-personas` revision `02471b26f7413b795702c3e60855d833694a2d64`. These released persona weights combine DPO weight 1.0 with introspective-SFT weight 0.25; they are not pure DPO-final checkpoints.
- OpenRLHF reference logic: commit `d21a99cd4c191fa0e461fd086dc59f8ceba0bba2`.

Adapter weight SHA-256 values:

- goodness: `25d02a4a6f3d6872121be1c2d816297bdd5a40da74704e5b79a31d7b4d49a5a0`
- humor: `24fc50a8acfa8b1e86cae6cba51d1bc87739ab17a4443ad61ee98484f39d8653`
- impulsiveness: `ccb04a37720494e0f35a41da06ffdff9eb1687a00b8b91303aab579c67d57f83`
- loving: `f8b5f7a9c4b647a5728467bf27548111ec7f11b068876ee68e5e86a9d633c456`
- mathematical: `2aa9c564ec029dff9d31a0855ce95883506a655b667fd70d7eaac4e52a756c43`
- nonchalance: `fd13980205eb8dce879db668663d6ff089321a5b1e15e370df80a4a564efb2a3`
- poeticism: `9b2a71209f140ced3f935fe41de544c63f0a7aba48e66df072f59c85d4350dd5`
- remorse: `9090089c0e21417230b46da51f0de455c81a249e76177c8c125f11b3f989c7e6`
- sarcasm: `7ca395c9bd5408157669751caa1832ba4a929f4cce8ae7acff766a8b9a16de89`
- sycophancy: `c913534bfb2a3dba09a465310273ddf4d2cc61c15b4145b752793b9e11888646`

## Tokenization and likelihood extraction

For each chosen or rejected two-message chat, the user-only prefix is rendered with `apply_chat_template(..., tokenize=False, add_generation_prompt=True)`. The full chat is rendered separately and must start with that exact prefix. Following the pinned OpenRLHF behavior, the response is sliced at the prefix character boundary, the recombined rendering has trailing newlines removed, and a space plus tokenizer EOS is appended if EOS is absent. Prompt and full rendering are tokenized with padding disabled, truncation disabled, and special-token insertion disabled. The last token ID is set to the tokenizer EOS ID. Inputs longer than 1,024 tokens fail rather than truncate.

vLLM is asked for prompt log-probabilities on those complete token IDs. For positions from the prompt-token boundary through EOS, the scorer looks up the actual input token and sums its log-probability. Missing tokens, misaligned output lengths, and nonfinite totals fail.

## Score and calibration orientation

For adapter/evaluation trait `C` on preference/training dataset `T`, positive score means the adapter increases the chosen-vs-rejected preference relative to base:

`[(adapter chosen - base chosen) - (adapter rejected - base rejected)]`.

The combined-length and separate-response variants divide as documented in the README. Native score matrices are `S[C,T]`: rows are adapter traits `C` and columns are preference datasets `T`. Behavioral matrices are `E[C,T]`: rows are evaluation traits `C` and columns are model training traits `T`, each relative to the row-specific base. Adapter trait `C` corresponds to evaluation trait `C`, while preference dataset `T` corresponds to training trait `T`, so the matrices align directly without a transpose. The public API exposes `align_score_to_behavior` and `score_behavior_correlation` so this shared orientation is checked explicitly.

The fixed primary is unnormalized sum, row-weighted mean, and population z-calibration (`ddof=0`) within each adapter row across all ten dataset columns. Sensitivities use combined-length or separate-response formulas, equal-prompt means, and leave-one-dataset-out calibration. Off-diagonal association excludes the ten matching-trait cells and reports Pearson or Spearman correlation over the remaining 90 aligned cells.
