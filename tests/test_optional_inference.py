from __future__ import annotations

import builtins
import importlib.util
import unittest
from pathlib import Path
from types import ModuleType
from unittest import mock

SCRIPT = Path(__file__).parents[1] / "scripts" / "score_oct_pairs.py"


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("score_oct_pairs", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load optional scoring script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OptionalInferenceTests(unittest.TestCase):
    def test_help_does_not_import_inference_dependencies(self) -> None:
        module = load_script()
        with self.assertRaises(SystemExit) as raised:
            module.main(["--help"])
        self.assertEqual(raised.exception.code, 0)

    def test_missing_inference_dependency_is_actionable(self) -> None:
        module = load_script()
        original_import = builtins.__import__

        def blocked(name: str, *args: object, **kwargs: object) -> object:
            if name == "transformers" or name.startswith("vllm"):
                raise ImportError("blocked optional dependency")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=blocked):
            with self.assertRaisesRegex(ImportError, "oct-llf.*vLLM"):
                module._inference_imports()


if __name__ == "__main__":
    unittest.main()
