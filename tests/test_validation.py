from __future__ import annotations

import json
import unittest
from pathlib import Path

import pyarrow as pa

from oct_llf.calibration import TRAITS
from oct_llf.validation import (
    SOURCE_DATA_REVISION,
    scan_repository_text,
    validate_bundle,
    validate_table,
)

ROOT = Path(__file__).resolve().parents[1]


def public_fixture(trait: str = "goodness") -> pa.Table:
    data = {
        "dataset_trait": [trait, trait],
        "source_revision": [SOURCE_DATA_REVISION, SOURCE_DATA_REVISION],
        "source_row_id": [0, 1],
        "prompt_hash": ["a" * 64, "b" * 64],
        "chosen_hash": ["c" * 64, "d" * 64],
        "rejected_hash": ["e" * 64, "f" * 64],
        "prompt_tokens": [4, 5],
        "chosen_response_tokens": [2, 4],
        "rejected_response_tokens": [3, 2],
        "base_chosen_logprob": [-5.0, -8.0],
        "base_rejected_logprob": [-6.0, -7.0],
    }
    for index, adapter in enumerate(TRAITS):
        chosen = [-4.0 + index, -7.0 + index]
        rejected = [-6.5, -6.5]
        data[f"{adapter}_chosen_logprob"] = chosen
        data[f"{adapter}_rejected_logprob"] = rejected
        chosen_shift = [chosen[0] + 5.0, chosen[1] + 8.0]
        rejected_shift = [rejected[0] + 6.0, rejected[1] + 7.0]
        summed = [chosen_shift[i] - rejected_shift[i] for i in range(2)]
        data[f"delta_sum_{adapter}"] = summed
        data[f"delta_lls_{adapter}"] = [summed[0] / 5, summed[1] / 6]
        data[f"delta_separate_{adapter}"] = [
            chosen_shift[0] / 2 - rejected_shift[0] / 3,
            chosen_shift[1] / 4 - rejected_shift[1] / 2,
        ]
    metadata = {b"oct_provenance": json.dumps({"contains_raw_text": False}).encode()}
    return pa.table(data).replace_schema_metadata(metadata)


class ValidationTests(unittest.TestCase):
    def test_public_schema_and_formula_roundtrip_fixture(self):
        result = validate_table(public_fixture(), "goodness")
        self.assertEqual(result["rows"], 2)
        self.assertEqual(result["formula_round_trip_max_abs_error"], 0.0)

    def test_full_bundled_data(self):
        result = validate_bundle(ROOT / "data", ROOT / "data" / "bundle_manifest.json")
        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["datasets_validated"], 10)
        self.assertEqual(result["total_rows"], 84803)
        self.assertEqual(result["total_bytes"], 47966184)
        self.assertEqual(result["formula_round_trip_max_abs_error"], 0.0)
        self.assertTrue(result["schema_identical"])

    def test_repository_is_public_only(self):
        result = scan_repository_text(ROOT)
        self.assertTrue(result["passed"], result["findings"])
        self.assertGreater(result["files_checked"], 10)


if __name__ == "__main__":
    unittest.main()
