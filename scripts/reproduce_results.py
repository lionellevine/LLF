#!/usr/bin/env python3
"""Reproduce LLF matrices, headline correlations, and deterministic resampling."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr

from oct_llf.calibration import (
    FORMULAS,
    TRAITS,
    WEIGHTINGS,
    aggregate_tables,
    calibrate,
    group_statistics,
    load_text_free_tables,
    score_arrays,
    summarize,
)
from oct_llf.scoring import sha256_file

TARGETS = {
    "combined_pearson": 0.104758806591627,
    "combined_spearman": 0.08656248328049311,
    "combined_bootstrap_low": 0.09074355424105954,
    "combined_bootstrap_high": 0.11810610551434357,
    "permutation_one_sided_p": 0.04911950880491195,
    "permutation_two_sided_p": 0.2590374096259037,
    "humor_column_pearson": 0.8069556769263959,
    "humor_column_spearman": 0.8,
    "humor_separate_spearman": -0.38333333333333336,
    "humor_sum_pearson": -0.6215649381632113,
}
BEHAVIOR_SHA256 = "4a8f90b634e635b08a32e5aaf7eee588fbe5738dcc9bab5a946b24a84bfaec3f"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_behavior_matrix(path: Path) -> np.ndarray:
    if sha256_file(path) != BEHAVIOR_SHA256:
        raise ValueError("behavior target SHA-256 does not match the pinned oct-v2 matrix")
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        label = reader.fieldnames[0] if reader.fieldnames else None
        if label != "evaluation_trait":
            raise ValueError("behavior matrix must start with evaluation_trait")
        rows = {row[label]: row for row in reader}
    if set(rows) != set(TRAITS):
        raise ValueError("behavior matrix must contain the ten canonical traits")
    matrix = np.asarray([[float(rows[row][column]) for column in TRAITS] for row in TRAITS])
    humor = TRAITS.index("humor")
    impulsiveness = TRAITS.index("impulsiveness")
    if not math.isclose(matrix[impulsiveness, humor], 128.3, abs_tol=1e-12):
        raise ValueError("behavior orientation sentinel E[impulsiveness,humor] is wrong")
    if not math.isclose(matrix[humor, impulsiveness], 118.4, abs_tol=1e-12):
        raise ValueError("behavior orientation sentinel E[humor,impulsiveness] is wrong")
    return matrix


def _bootstrap_aggregates(
    group_sums: np.ndarray,
    group_counts: np.ndarray,
    replicates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    groups = len(group_counts)
    multiplicity = rng.multinomial(groups, np.full(groups, 1.0 / groups), size=replicates)
    numerator = np.einsum("bg,gfa->bfa", multiplicity, group_sums, optimize=True)
    row = numerator / (multiplicity @ group_counts)[:, None, None]
    group_means = group_sums / group_counts[:, None, None]
    prompt = np.einsum("bg,gfa->bfa", multiplicity, group_means, optimize=True) / groups
    return np.stack((row, prompt), axis=1)


def _batch_pearson(scores: np.ndarray, target: np.ndarray) -> np.ndarray:
    mask = np.logical_not(np.eye(len(TRAITS), dtype=bool))
    x = scores[..., mask]
    y = target[mask]
    x_centered = x - x.mean(axis=-1, keepdims=True)
    y_centered = y - y.mean()
    return np.sum(x_centered * y_centered, axis=-1) / np.sqrt(
        np.sum(np.square(x_centered), axis=-1) * np.sum(np.square(y_centered))
    )


def _resample(
    tables: dict[str, Any],
    primary: np.ndarray,
    behavior: np.ndarray,
    bootstrap_replicates: int,
    permutation_draws: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(42)
    bootstrap_raw = np.empty(
        (bootstrap_replicates, len(FORMULAS), len(WEIGHTINGS), len(TRAITS), len(TRAITS))
    )
    for dataset_index, trait in enumerate(TRAITS):
        scores, prompt_hashes = score_arrays(tables[trait])
        group_sums, group_counts = group_statistics(scores, prompt_hashes)
        boot = _bootstrap_aggregates(group_sums, group_counts, bootstrap_replicates, rng)
        bootstrap_raw[..., dataset_index] = boot.transpose(0, 2, 1, 3)
    bootstrap_calibrated, _, _ = calibrate(bootstrap_raw, "all")
    bootstrap_primary = bootstrap_calibrated[:, 0, 0]
    bootstrap_values = _batch_pearson(bootstrap_primary, behavior)

    mask = np.logical_not(np.eye(len(TRAITS), dtype=bool))
    observed = float(pearsonr(primary[mask], behavior[mask]).statistic)
    null = np.empty(permutation_draws)
    for draw in range(permutation_draws):
        permuted = behavior[rng.permutation(len(TRAITS)), :]
        null[draw] = float(pearsonr(primary[mask], permuted[mask]).statistic)
    one_sided = float((1 + np.sum(null >= observed)) / (len(null) + 1))
    two_sided = float((1 + np.sum(np.abs(null) >= abs(observed))) / (len(null) + 1))
    return {
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_95_interval": [
            float(np.percentile(bootstrap_values, 2.5)),
            float(np.percentile(bootstrap_values, 97.5)),
        ],
        "permutation_draws": permutation_draws,
        "permutation_one_sided_p": one_sided,
        "permutation_two_sided_p": two_sided,
        "seed": 42,
    }


def reproduce(
    data_dir: Path,
    behavior_path: Path,
    output_dir: Path,
    bootstrap_replicates: int = 1000,
    permutation_draws: int = 100000,
) -> dict[str, Any]:
    if bootstrap_replicates < 1 or permutation_draws < 1:
        raise ValueError("resampling counts must be positive")
    matrices_dir = output_dir / "matrices"
    summarize(data_dir, matrices_dir)

    expected_hashes = json.loads(
        (behavior_path.parent / "matrix_hashes.json").read_text(encoding="utf-8")
    )
    observed_hashes = {name: sha256_file(matrices_dir / name) for name in expected_hashes}
    mismatches = {
        name: {"expected": expected_hashes[name], "observed": observed_hashes[name]}
        for name in expected_hashes
        if observed_hashes[name] != expected_hashes[name]
    }
    if mismatches:
        raise AssertionError(f"matrix SHA-256 mismatch: {mismatches}")

    tables = load_text_free_tables(data_dir)
    raw = aggregate_tables(tables)
    primary, _, _ = calibrate(raw[FORMULAS.index("sum"), WEIGHTINGS.index("row")], "all")
    behavior = load_behavior_matrix(behavior_path)
    off_diagonal = np.logical_not(np.eye(len(TRAITS), dtype=bool))
    combined_pearson = float(pearsonr(primary[off_diagonal], behavior[off_diagonal]).statistic)
    combined_spearman = float(spearmanr(primary[off_diagonal], behavior[off_diagonal]).statistic)

    humor = TRAITS.index("humor")
    off_humor = np.not_equal(np.arange(len(TRAITS)), humor)
    humor_column_pearson = float(
        pearsonr(primary[off_humor, humor], behavior[off_humor, humor]).statistic
    )
    humor_column_spearman = float(
        spearmanr(primary[off_humor, humor], behavior[off_humor, humor]).statistic
    )
    humor_primary_order = [
        TRAITS[index]
        for index in np.where(off_humor)[0][np.argsort(primary[off_humor, humor])[::-1]]
    ]

    separate = raw[FORMULAS.index("separate"), WEIGHTINGS.index("row")]
    summed = raw[FORMULAS.index("sum"), WEIGHTINGS.index("row")]
    humor_separate_spearman = float(
        spearmanr(separate[off_humor, humor], behavior[off_humor, humor]).statistic
    )
    humor_sum_pearson = float(
        pearsonr(summed[off_humor, humor], behavior[off_humor, humor]).statistic
    )
    humor_separate_order = [
        TRAITS[index]
        for index in np.where(off_humor)[0][np.argsort(separate[off_humor, humor])[::-1]]
    ]

    resampling = _resample(
        tables,
        primary,
        behavior,
        bootstrap_replicates,
        permutation_draws,
    )
    combined = {
        "cells": 90,
        "pearson": combined_pearson,
        "spearman": combined_spearman,
        "humor_column": {
            "cells": 9,
            "pearson": humor_column_pearson,
            "spearman": humor_column_spearman,
            "impulsiveness_rank": 1 + humor_primary_order.index("impulsiveness"),
        },
        **resampling,
    }
    humor_result = {
        "rows": 6806,
        "prompt_groups": 1752,
        "separate_response_off_target_spearman": humor_separate_spearman,
        "unnormalized_sum_off_target_pearson": humor_sum_pearson,
        "impulsiveness_rank": 1 + humor_separate_order.index("impulsiveness"),
    }

    deterministic_checks = [
        (combined_pearson, TARGETS["combined_pearson"], 1e-9),
        (combined_spearman, TARGETS["combined_spearman"], 1e-12),
        (humor_column_pearson, TARGETS["humor_column_pearson"], 1e-9),
        (humor_column_spearman, TARGETS["humor_column_spearman"], 1e-12),
        (humor_separate_spearman, TARGETS["humor_separate_spearman"], 1e-12),
        (humor_sum_pearson, TARGETS["humor_sum_pearson"], 1e-6),
    ]
    if any(
        not math.isclose(value, target, abs_tol=tolerance)
        for value, target, tolerance in deterministic_checks
    ):
        raise AssertionError("one or more fixed headline targets did not reproduce")
    if combined["humor_column"]["impulsiveness_rank"] != 3:
        raise AssertionError("combined humor-column impulsiveness rank did not reproduce")
    if humor_result["impulsiveness_rank"] not in (7, 8):
        raise AssertionError("single-humor impulsiveness rank did not reproduce")

    full_resampling = bootstrap_replicates == 1000 and permutation_draws == 100000
    if full_resampling:
        stochastic_checks = [
            (resampling["bootstrap_95_interval"][0], TARGETS["combined_bootstrap_low"], 1e-12),
            (resampling["bootstrap_95_interval"][1], TARGETS["combined_bootstrap_high"], 1e-12),
            (resampling["permutation_one_sided_p"], TARGETS["permutation_one_sided_p"], 1e-15),
            (resampling["permutation_two_sided_p"], TARGETS["permutation_two_sided_p"], 1e-15),
        ]
        if any(
            not math.isclose(value, target, abs_tol=tolerance)
            for value, target, tolerance in stochastic_checks
        ):
            raise AssertionError("full deterministic resampling targets did not reproduce")

    _write_json(output_dir / "combined_stage_matrix" / "summary.json", combined)
    _write_json(output_dir / "humor_single_dataset" / "summary.json", humor_result)
    result = {
        "status": "passed",
        "mode": "full" if full_resampling else "smoke",
        "behavior_target_sha256": BEHAVIOR_SHA256,
        "matrix_hashes": observed_hashes,
        "combined_stage_matrix": combined,
        "humor_single_dataset": humor_result,
    }
    _write_json(output_dir / "reproduction_summary.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument(
        "--behavior",
        type=Path,
        default=Path("data/reference/behavior_oct_v2.csv"),
    )
    parser.add_argument("--output", type=Path, default=Path("results/reproduced"))
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--permutation-draws", type=int, default=100000)
    args = parser.parse_args()
    result = reproduce(
        args.data,
        args.behavior,
        args.output,
        args.bootstrap_replicates,
        args.permutation_draws,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
