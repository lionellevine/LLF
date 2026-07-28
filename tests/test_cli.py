from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from oct_llf.cli import main


class CliTests(unittest.TestCase):
    def test_score_formulas_json(self):
        output = io.StringIO()
        with redirect_stdout(output):
            status = main(["score-formulas", "-8", "-10", "4", "-10", "-13", "3"])
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(output.getvalue())["sum"], -1.0)

    def test_validate_exit_statuses(self):
        with patch("oct_llf.cli.validate_bundle", return_value={"passed": True}):
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["validate-data", "data"]), 0)
        with patch("oct_llf.cli.validate_bundle", return_value={"passed": False}):
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["validate-data", "data"]), 1)

    def test_summarize_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            expected = {"files_written": 1}
            with patch("oct_llf.cli.summarize", return_value=expected) as mocked:
                with redirect_stdout(io.StringIO()):
                    status = main(["summarize", "data", "--output", directory])
            self.assertEqual(status, 0)
            mocked.assert_called_once_with(Path("data"), Path(directory))


if __name__ == "__main__":
    unittest.main()
