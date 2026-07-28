"""Aggregation, calibration, sensitivity, and resampling for OCT score tables."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import pearsonr, spearmanr

from .scoring import derived_score_arrays

TRAITS = (
    "goodness",
    "humor",
    "impulsiveness",
    "loving",
    "mathematical",
    "nonchalance",
    "poeticism",
    "remorse",
    "sarcasm",
    "sycophancy",
)
FORMULAS = ("sum", "lls", "separate")
WEIGHTINGS = ("row", "prompt")
CALIBRATIONS = ("all", "leave_one_dataset_out")


def load_text_free_tables(data_dir: str | Path) -> dict[str, pa.Table]:
    """Load the ten named Parquet tables without accepting extra/missing datasets."""

    root = Path(data_dir)
    parquet_root = root / "logprobs" if (root / "logprobs").is_dir() else root
    found = {path.stem: path for path in parquet_root.glob("*.parquet")}
    if set(found) != set(TRAITS):
        raise ValueError(f"expected datasets {list(TRAITS)}, found {sorted(found)}")
    return {trait: pq.read_table(found[trait]) for trait in TRAITS}


def score_arrays(table: pa.Table, traits: Sequence[str] = TRAITS) -> tuple[np.ndarray, np.ndarray]:
    """Return scores shaped ``(row, formula, adapter)`` and prompt hashes."""

    required = {
        "prompt_hash",
        "chosen_response_tokens",
        "rejected_response_tokens",
        "base_chosen_logprob",
        "base_rejected_logprob",
    }
    for trait in traits:
        required.update({f"{trait}_chosen_logprob", f"{trait}_rejected_logprob"})
    missing = required - set(table.column_names)
    if missing:
        raise ValueError(f"table is missing required columns: {sorted(missing)}")
    values = np.empty((table.num_rows, len(FORMULAS), len(traits)), dtype=float)
    common = {
        "base_chosen": table["base_chosen_logprob"].to_numpy(),
        "chosen_length": table["chosen_response_tokens"].to_numpy(),
        "base_rejected": table["base_rejected_logprob"].to_numpy(),
        "rejected_length": table["rejected_response_tokens"].to_numpy(),
    }
    for index, trait in enumerate(traits):
        derived = derived_score_arrays(
            table[f"{trait}_chosen_logprob"].to_numpy(),
            common["base_chosen"],
            common["chosen_length"],
            table[f"{trait}_rejected_logprob"].to_numpy(),
            common["base_rejected"],
            common["rejected_length"],
        )
        for formula_index, formula in enumerate(FORMULAS):
            values[:, formula_index, index] = derived[formula]
    hashes = np.asarray(table["prompt_hash"].to_pylist(), dtype=object)
    return values, hashes


def group_statistics(
    scores: np.ndarray, prompt_hashes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse rows to prompt-group sums and counts."""

    if len(scores) != len(prompt_hashes) or len(scores) == 0:
        raise ValueError("scores and nonempty prompt hashes must align")
    _, inverse = np.unique(prompt_hashes, return_inverse=True)
    counts = np.bincount(inverse).astype(float)
    sums = np.zeros((len(counts), *scores.shape[1:]), dtype=float)
    np.add.at(sums, inverse, scores)
    return sums, counts


def aggregate_scores(scores: np.ndarray, prompt_hashes: np.ndarray) -> np.ndarray:
    """Return row-weighted and equal-prompt means as ``(weighting, formula, adapter)``."""

    group_sums, group_counts = group_statistics(scores, prompt_hashes)
    return np.stack((scores.mean(axis=0), (group_sums / group_counts[:, None, None]).mean(axis=0)))


def aggregate_tables(tables: Mapping[str, pa.Table]) -> np.ndarray:
    """Build ``raw[formula, weighting, adapter, dataset]`` in canonical orientation."""

    if set(tables) != set(TRAITS):
        raise ValueError("tables must contain exactly the ten canonical traits")
    raw = np.empty((len(FORMULAS), len(WEIGHTINGS), len(TRAITS), len(TRAITS)))
    for dataset_index, dataset_trait in enumerate(TRAITS):
        scores, hashes = score_arrays(tables[dataset_trait])
        observed = aggregate_scores(scores, hashes)
        raw[:, :, :, dataset_index] = observed.transpose(1, 0, 2)
    return raw


def calibrate(matrix: np.ndarray, scheme: str = "all") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Population-z calibrate each adapter row over datasets.

    The last two axes must be adapter/training rows then preference-dataset columns.
    """

    values = np.asarray(matrix, dtype=float)
    if values.shape[-2:] != (len(TRAITS), len(TRAITS)):
        raise ValueError(f"last axes must be adapter,dataset = (10,10), got {values.shape}")
    if scheme == "all":
        mean = values.mean(axis=-1, keepdims=True)
        std = values.std(axis=-1, ddof=0, keepdims=True)
        parameters_mean, parameters_std = mean[..., 0], std[..., 0]
    elif scheme == "leave_one_dataset_out":
        n = values.shape[-1] - 1
        mean = (values.sum(axis=-1, keepdims=True) - values) / n
        variance = (
            np.square(values).sum(axis=-1, keepdims=True) - np.square(values)
        ) / n - np.square(mean)
        std = np.sqrt(np.maximum(variance, 0.0))
        parameters_mean, parameters_std = mean, std
    else:
        raise ValueError(f"unknown calibration scheme: {scheme}")
    if np.any(std <= 0) or not np.all(np.isfinite(std)):
        raise ValueError("calibration has zero or nonfinite population standard deviation")
    return (values - mean) / std, parameters_mean, parameters_std


def align_score_to_behavior(score: np.ndarray) -> np.ndarray:
    """Return scores in the already-matching behavioral matrix orientation.

    Native score matrices are ``score[adapter_trait, preference_dataset]`` or
    ``S[C,T]``. Behavioral matrices are ``behavior[evaluation_trait,
    training_trait]`` or ``E[C,T]``. Adapter trait C corresponds to evaluation
    trait C, and preference dataset T corresponds to training trait T, so no
    transpose is required.
    """

    values = np.asarray(score, dtype=float)
    if values.shape != (10, 10):
        raise ValueError("native score matrix must be 10x10")
    return values.copy()


def off_diagonal_correlation(
    score: np.ndarray, target: np.ndarray, method: str = "pearson"
) -> float:
    """Correlate two already-aligned 10×10 matrices over 90 off-diagonal cells."""

    score, target = np.asarray(score), np.asarray(target)
    if score.shape != (10, 10) or target.shape != (10, 10):
        raise ValueError("score and target must both be 10x10")
    mask = ~np.eye(10, dtype=bool)
    if method == "pearson":
        return float(pearsonr(score[mask], target[mask]).statistic)
    if method == "spearman":
        return float(spearmanr(score[mask], target[mask]).statistic)
    raise ValueError("method must be pearson or spearman")


def prompt_group_bootstrap(
    scores: np.ndarray,
    prompt_hashes: np.ndarray,
    replicates: int,
    seed: int = 42,
) -> np.ndarray:
    """Bootstrap prompt groups and return both weighting means per replicate."""

    if replicates < 1:
        raise ValueError("replicates must be positive")
    sums, counts = group_statistics(scores, prompt_hashes)
    groups = len(counts)
    rng = np.random.default_rng(seed)
    multiplicity = rng.multinomial(groups, np.full(groups, 1 / groups), size=replicates)
    numerator = np.einsum("bg,gfa->bfa", multiplicity, sums, optimize=True)
    row = numerator / (multiplicity @ counts)[:, None, None]
    prompt = (
        np.einsum("bg,gfa->bfa", multiplicity, sums / counts[:, None, None], optimize=True) / groups
    )
    return np.stack((row, prompt), axis=1)


def score_behavior_correlation(
    native_score: np.ndarray, behavior: np.ndarray, method: str = "pearson"
) -> float:
    """Correlate directly aligned adapter-by-dataset and evaluation-by-training matrices."""

    return off_diagonal_correlation(align_score_to_behavior(native_score), behavior, method)


def evaluation_row_block_permutation(
    native_score: np.ndarray,
    behavior: np.ndarray,
    draws: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Permute behavior evaluation rows against the directly aligned score matrix."""

    if draws < 1:
        raise ValueError("draws must be positive")
    aligned_score = align_score_to_behavior(native_score)
    behavior = np.asarray(behavior, dtype=float)
    if behavior.shape != (10, 10):
        raise ValueError("behavior matrix must be evaluation-by-training with shape 10x10")
    rng = np.random.default_rng(seed)
    values, permutations = np.empty(draws), np.empty((draws, 10), dtype=np.int8)
    for index in range(draws):
        permutation = rng.permutation(10)
        permutations[index] = permutation
        values[index] = off_diagonal_correlation(aligned_score, behavior[permutation, :])
    return values, permutations


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(text)
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def _matrix_csv(matrix: np.ndarray) -> str:
    lines = [",".join(("adapter_trait", *TRAITS))]
    for trait, row in zip(TRAITS, matrix, strict=True):
        lines.append(",".join((trait, *(f"{value:.12g}" for value in row))))
    return "\n".join(lines) + "\n"


def summarize(data_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Atomically write raw and calibrated sensitivity matrices plus primary parameters."""

    raw = aggregate_tables(load_text_free_tables(data_dir))
    output = Path(output_dir)
    files: list[str] = []
    calibrated: dict[str, np.ndarray] = {}
    for f, formula in enumerate(FORMULAS):
        for w, weighting in enumerate(WEIGHTINGS):
            name = f"raw_{formula}_{weighting}.csv"
            _atomic_text(output / name, _matrix_csv(raw[f, w]))
            files.append(name)
    for scheme in CALIBRATIONS:
        calibrated[scheme], _, _ = calibrate(raw, scheme)
        for f, formula in enumerate(FORMULAS):
            for w, weighting in enumerate(WEIGHTINGS):
                name = f"calibrated_{formula}_{weighting}_{scheme}.csv"
                _atomic_text(output / name, _matrix_csv(calibrated[scheme][f, w]))
                files.append(name)
    primary_calibrated, means, stds = calibrate(raw[0, 0], "all")
    params = {
        trait: {"mean": float(means[index]), "population_sd": float(stds[index])}
        for index, trait in enumerate(TRAITS)
    }
    _atomic_text(output / "primary_calibrated_sum_row_all.csv", _matrix_csv(primary_calibrated))
    _atomic_text(
        output / "calibration_parameters.json", json.dumps(params, indent=2, sort_keys=True) + "\n"
    )
    sensitivity = {
        "orientation": "rows=adapter/training trait; columns=preference dataset",
        "primary": {"formula": "sum", "weighting": "row", "calibration": "all"},
        "matrices": files,
    }
    _atomic_text(
        output / "sensitivity.json", json.dumps(sensitivity, indent=2, sort_keys=True) + "\n"
    )
    return {
        "files_written": len(files) + 3,
        "primary": sensitivity["primary"],
        "parameters": params,
    }
