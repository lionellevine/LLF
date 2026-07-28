from __future__ import annotations

import unittest

import numpy as np
import pyarrow as pa

from oct_llf.calibration import (
    FORMULAS,
    TRAITS,
    aggregate_tables,
    align_score_to_behavior,
    calibrate,
    evaluation_row_block_permutation,
    off_diagonal_correlation,
    prompt_group_bootstrap,
    score_behavior_correlation,
)


def synthetic_table(dataset_index: int) -> pa.Table:
    data = {
        "prompt_hash": ["a" * 64, "b" * 64],
        "chosen_response_tokens": [2, 2],
        "rejected_response_tokens": [3, 3],
        "base_chosen_logprob": [-5.0, -5.0],
        "base_rejected_logprob": [-6.0, -6.0],
    }
    for adapter_index, trait in enumerate(TRAITS):
        value = 100.0 * adapter_index + dataset_index
        data[f"{trait}_chosen_logprob"] = [-5.0 + value, -5.0 + value]
        data[f"{trait}_rejected_logprob"] = [-6.0, -6.0]
    return pa.table(data)


class CalibrationTests(unittest.TestCase):
    def test_primary_and_sensitivity_orientation(self):
        tables = {trait: synthetic_table(index) for index, trait in enumerate(TRAITS)}
        raw = aggregate_tables(tables)
        self.assertEqual(raw.shape, (3, 2, 10, 10))
        self.assertEqual(raw[FORMULAS.index("sum"), 0, 3, 7], 307.0)
        self.assertEqual(raw[FORMULAS.index("lls"), 0, 3, 7], 307.0 / 5.0)
        self.assertEqual(raw[FORMULAS.index("separate"), 0, 3, 7], 307.0 / 2.0)
        primary, mean, std = calibrate(raw[0, 0], "all")
        self.assertTrue(np.allclose(primary.mean(axis=1), 0))
        self.assertTrue(np.allclose(primary.std(axis=1, ddof=0), 1))
        self.assertEqual(mean.shape, (10,))
        self.assertEqual(std.shape, (10,))
        loo, loo_mean, loo_std = calibrate(raw[0, 0], "leave_one_dataset_out")
        self.assertEqual(loo.shape, (10, 10))
        self.assertEqual(loo_mean.shape, (10, 10))
        self.assertEqual(loo_std.shape, (10, 10))

    def test_behavior_alignment_preserves_matching_axes(self):
        native_score = np.random.default_rng(9).normal(size=(10, 10))
        behavior = native_score * 2.5 + 7.0
        self.assertTrue(np.array_equal(align_score_to_behavior(native_score), native_score))
        self.assertAlmostEqual(score_behavior_correlation(native_score, behavior), 1.0)
        self.assertAlmostEqual(off_diagonal_correlation(native_score, behavior), 1.0)

    def test_prompt_group_bootstrap_is_deterministic(self):
        scores = np.arange(24, dtype=float).reshape(6, 2, 2)
        hashes = np.asarray(["a", "a", "b", "c", "c", "c"], dtype=object)
        first = prompt_group_bootstrap(scores, hashes, 20, seed=7)
        second = prompt_group_bootstrap(scores, hashes, 20, seed=7)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(first.shape, (20, 2, 2, 2))
        self.assertFalse(np.array_equal(first[:, 0], first[:, 1]))

    def test_evaluation_row_block_permutation(self):
        score = np.arange(100, dtype=float).reshape(10, 10)
        target = np.square(score)
        values_a, permutations_a = evaluation_row_block_permutation(score, target, 12, seed=4)
        values_b, permutations_b = evaluation_row_block_permutation(score, target, 12, seed=4)
        self.assertTrue(np.array_equal(permutations_a, permutations_b))
        self.assertTrue(np.array_equal(values_a, values_b))
        for permutation in permutations_a:
            self.assertEqual(set(permutation.tolist()), set(range(10)))


if __name__ == "__main__":
    unittest.main()
