# LLF

LLF is a small, public Python library for validated log-probability scoring and calibration of Open Character Training (OCT) preference pairs. The distribution is `oct-llf`; import it as `oct_llf`.

The repository has two distinct scopes:

1. **Precomputed-data analysis (CPU):** validate the ten bundled text-free Parquet files and reproduce raw, calibrated, and sensitivity matrices. This is the default, fully bundled path.
2. **Optional GPU scoring:** render preference responses with the pinned OpenRLHF-compatible boundary rule and extract response-token totals from vLLM prompt log-probabilities. Model weights, adapters, and source text are not bundled; `scripts/score_oct_pairs.py` is the text-free GPU runner.

## Quickstart

```bash
python -m pip install -e .
oct-llf validate-data data --manifest data/bundle_manifest.json
oct-llf summarize data --output outputs
python scripts/reproduce_results.py --data data --output results/reproduced
```

Compute one pair's formulas directly:

```bash
oct-llf score-formulas -8 -10 4 -10 -13 3
```

For optional inference, start in an environment with the matching CUDA runtime, install the tested torch/vLLM builds, then install the lightweight inference extra:

```bash
python -m pip install 'torch==2.11.0'
python -m pip install 'https://github.com/vllm-project/vllm/releases/download/v0.25.1/vllm-0.25.1%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl'
python -m pip install -e '.[inference]'
```

The validated inference stack was torch 2.11.0, transformers 5.12.1, vLLM 0.25.1+cu129, and peft 0.19.1. Torch and vLLM remain explicit platform installs because the tested local-version CUDA wheel is not a normal PyPI release and should not be downloaded during CPU-only analysis or CI. Hardware and backend differences can alter primitive log-probabilities; see [validation history](docs/VALIDATION.md).

Score a public OCT-style JSONL and emit only hashes, token counts, response totals, and derived formulas:

```bash
python scripts/score_oct_pairs.py pairs.jsonl outputs/humor.jsonl \
  --model Qwen/Qwen2.5-7B-Instruct \
  --revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --adapter maius/qwen-2.5-7b-it-personas \
  --adapter-revision 02471b26f7413b795702c3e60855d833694a2d64 \
  --adapter-name humor
```

The script fails unless a Hub adapter has an immutable revision. It was smoke-tested for dependency isolation and input/output safety during export; the source experiments, not this CPU-only export, provide the GPU numerical validation record.

## Recorded result

From the bundled 84,803 text-free rows and pinned `oct-v2` target, the calibrated unnormalized sum has Pearson **0.104758806591627** across 90 off-diagonal cells. The humor training column has Pearson **0.8069556769263959**. On the 6,806-row humor preference dataset alone, the separate-response off-target Spearman is **-0.38333333333333336** and the unnormalized Pearson is **-0.6215649381632113**.

These are correlations, not causal effects. They apply to Qwen2.5-7B-Instruct and the ten combined public persona adapters at the pinned revisions below. Reproduce the exact point estimates, 1,000-bootstrap interval, 100,000-permutation p-values, and 18 matrix hashes with `python scripts/reproduce_results.py`; stored outputs are under `results/`.

## Scores

Let `A_c, A_r` be adapter total log-probabilities for chosen and rejected responses; `B_c, B_r` the corresponding base totals; and `n_c, n_r > 0` their response-token counts. Define shifts `s_c=A_c-B_c` and `s_r=A_r-B_r`.

- **Sum (fixed primary):** `D_sum = s_c - s_r`
- **Combined-length / LLS:** `D_lls = D_sum / (n_c + n_r)`
- **Separate-response:** `D_sep = s_c/n_c - s_r/n_r`

Positive means the adapter moves relative preference toward the chosen response. Native score matrices are `S[C,T]`: **rows = adapter/evaluation trait C** and **columns = preference/training dataset T**. Behavioral matrices are `E[C,T]` on those same axes, so cellwise association is direct; `align_score_to_behavior` and `score_behavior_correlation` enforce the shared 10×10 orientation without transposing. The fixed primary is unnormalized sum, row-weighted mean, followed by population z-calibration (`ddof=0`) within each adapter across all ten datasets. Leave-one-dataset-out calibration, equal-prompt weighting, and both normalized formulas are sensitivity variants, not post-hoc replacements.

The ten public adapters evaluated here are the released combined persona weights produced by DPO weight 1.0 plus introspective-SFT weight 0.25. They are **not** pure DPO-final checkpoints; a future pure-DPO matrix is a new dataset version, not a relabeling of these results.

## Python API

```python
from oct_llf import openrlhf_encode_response, score_formulas, validate_bundle

result = validate_bundle("data", "data/bundle_manifest.json")
assert result["passed"]
```

`openrlhf_encode_response` requires a Transformers-compatible tokenizer but imports no inference dependency at module import time. `extract_vllm_response_logprob` accepts vLLM output objects and sums only actual response-token log-probabilities. Encoding fails rather than truncating sequences beyond the configured maximum.

## Repository map

- `src/oct_llf/`: scoring, calibration, validation, and CLI APIs
- `scripts/score_oct_pairs.py`: optional GPU JSONL-to-text-free-totals entry point
- `scripts/build_score_matrix.py` and `scripts/reproduce_results.py`: CPU matrix and headline-result replay
- `data/logprobs/`: ten canonical text-free Parquets; not included in package distributions
- `data/reference/`: the fixed analysis specification and reference matrix
- `docs/`: schema, provenance, validation boundaries, data card, and [extension guide](docs/EXTENDING.md)
- `tests/`: 15 archive tests plus optional-inference export tests

## Related work and source experiments

- Maiya, Bartsch, Lambert, and Hubinger, [*Open Character Training*](https://arxiv.org/abs/2511.01689)
- Chang, Piff, Sana, Li, and Levine, [*EigenBench*](https://arxiv.org/abs/2509.01938)
- [*Side Effects of Character Training: Quantifying Cross-Constitution Drift in LLMs*](https://openreview.net/forum?id=oh9CqCyxSc)
- Aden-Ali et al., [*Subliminal Effects in Your Data: A General Mechanism via Log-Linearity*](https://arxiv.org/abs/2602.04863)
- Silico scorer/validation experiment `exp_01kxn6a6e3enpb98n5byw7rja1`
- Silico ten-dataset calibration experiment `exp_01kxtjv8cbe9qt61nrx32xq318`

## Scope and caveats

- The bundle supports reproducible scoring, aggregation, calibration, and resampling. It does not establish that a direction causes behavior, nor does it redistribute model weights or source preference text.
- The public Parquet bundle is text-free. It contains deterministic prompt/chosen/rejected hashes, which may enable linkage to rows in a public source dataset. Treat hashes as pseudonymous identifiers, not an anonymity guarantee.
- Exact token parity with the pinned OpenRLHF logic passed. A historical primitive Transformers-vLLM absolute agreement gate failed and was never retroactively passed. A later, different within-vLLM derived-delta repeatability gate passed. These claims are not interchangeable; details are in `docs/VALIDATION.md`.
- The immutable Parquet files are repository research data and are deliberately excluded from both wheel and source-distribution payloads.

## Attribution

This repository was created by [Silico](https://www.goodfire.ai/silico) at the direction of [Lionel Levine](https://lionellevine.github.io). Attentive readers will notice two of Silico's tells: a deep fondness for the word "smoke" (nothing here ships un-smoke-tested), and a habit of reporting a frankly ridiculous number of significant figures — Pearson **0.104758806591627**, in case the fifteenth digit changes your conclusions. Science fiction promised us robots that recite long strings of digits nobody asked for; another trope realized!

## Data terms

The code is MIT licensed. The data is **not relicensed**. The source dataset revision is `2577813a6a435d21051c0548ff2f29dc897212d7`; its stated CC BY-NC-SA/non-commercial research terms apply unless an underlying source is stricter. Review `NOTICE-DATA` and `docs/DATA_CARD.md` before reuse.
