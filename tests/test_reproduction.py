from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "reproduce_results.py"


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("reproduce_results", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load reproduction script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReproductionTests(unittest.TestCase):
    def test_headline_metrics_and_matrix_hashes_reproduce(self) -> None:
        module = load_script()
        with tempfile.TemporaryDirectory() as temporary:
            result = module.reproduce(
                ROOT / "data",
                ROOT / "data" / "reference" / "behavior_oct_v2.csv",
                Path(temporary),
                bootstrap_replicates=10,
                permutation_draws=100,
            )
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["mode"], "smoke")
        combined = result["combined_stage_matrix"]
        humor = result["humor_single_dataset"]
        self.assertAlmostEqual(combined["pearson"], 0.104758806591627, places=9)
        self.assertAlmostEqual(combined["humor_column"]["pearson"], 0.8069556769263959, places=9)
        self.assertAlmostEqual(
            humor["separate_response_off_target_spearman"],
            -0.38333333333333336,
            places=12,
        )
        self.assertEqual(len(result["matrix_hashes"]), 18)


if __name__ == "__main__":
    unittest.main()
