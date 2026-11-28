from contextlib import redirect_stderr
from io import StringIO
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import matplotlib
import pandas as pd

from icu_forecast import OCCUPANCY, main, write_results

matplotlib.use("Agg")


class ExportTests(unittest.TestCase):
    def setUp(self):
        scratch = TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        self.input = self.root / "input.csv"
        self.input.write_bytes(b"original input")
        self.output = self.root / "results"
        self.output.mkdir()
        self.csv = self.output / "icu_breach_probabilities.csv"
        self.chart = self.output / "SHAP_features.png"
        self.stage = self.output / ".icu-forecast-export"
        self.predictions = pd.DataFrame({"prob_breach_7d": [0.25, 0.8]})
        self.importance = pd.Series({OCCUPANCY: 0.42})

    def export(self):
        write_results(self.input, self.output, self.predictions, self.importance)

    def test_cli_rejects_aliased_input_before_reading_or_training(self):
        self.chart.write_bytes(b"previous figure")
        os.link(self.input, self.csv)
        for source in [self.csv, self.input]:
            with self.subTest(source=source.name), redirect_stderr(StringIO()) as diagnostic:
                with patch("icu_forecast.read_snapshot", side_effect=AssertionError("read before validation")):
                    with self.assertRaises(SystemExit) as error:
                        main(["--csv", str(source), "--output-dir", str(self.output)])
                self.assertEqual(error.exception.code, 2)
                self.assertIn("distinct files", diagnostic.getvalue())
        self.assertEqual(self.input.read_bytes(), b"original input")
        self.assertEqual(self.csv.read_bytes(), b"original input")
        self.assertEqual(self.chart.read_bytes(), b"previous figure")
        self.assertFalse(self.stage.exists())

    def test_output_pair_cannot_share_a_file_identity(self):
        self.csv.write_bytes(b"earlier output")
        os.link(self.csv, self.chart)
        with self.assertRaisesRegex(ValueError, "distinct files"):
            self.export()
        self.assertEqual(self.chart.read_bytes(), b"earlier output")
        self.assertTrue(self.csv.samefile(self.chart))
        self.assertFalse(self.stage.exists())

    def test_plot_failure_preserves_both_prior_outputs(self):
        self.csv.write_bytes(b"old predictions")
        self.chart.write_bytes(b"old figure")
        with patch("matplotlib.figure.Figure.savefig", side_effect=OSError("plot write failed")):
            with self.assertRaisesRegex(OSError, "plot write failed"):
                self.export()
        self.assertEqual(self.csv.read_bytes(), b"old predictions")
        self.assertEqual(self.chart.read_bytes(), b"old figure")
        self.assertEqual(self.input.read_bytes(), b"original input")
        self.assertFalse(self.stage.exists())

    def test_directory_at_chart_path_leaves_previous_csv_intact(self):
        self.csv.write_bytes(b"earlier table")
        self.chart.mkdir()
        with self.assertRaisesRegex(ValueError, "regular file"):
            self.export()
        self.assertTrue(self.chart.is_dir())
        self.assertEqual(self.csv.read_bytes(), b"earlier table")
        self.assertFalse(self.stage.exists())

    def test_chart_install_failure_restores_csv_or_removes_a_new_csv(self):
        original_replace = Path.replace
        self.chart.write_bytes(b"old figure")

        def locked_chart(path, target):
            if target == self.chart:
                raise PermissionError("chart is open")
            return original_replace(path, target)

        for previous in (None, b"old predictions"):
            with self.subTest(previous=previous):
                if previous is not None:
                    self.csv.write_bytes(previous)
                with patch.object(Path, "replace", locked_chart), self.assertRaisesRegex(PermissionError, "chart is open"):
                    self.export()
                self.assertEqual(self.csv.read_bytes() if self.csv.exists() else None, previous)
                self.assertEqual(self.chart.read_bytes(), b"old figure")
                self.assertEqual(self.input.read_bytes(), b"original input")
                self.assertFalse(self.stage.exists())

        self.export()
        pd.testing.assert_frame_equal(pd.read_csv(self.csv), self.predictions)
        self.assertTrue(self.chart.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertEqual(set(self.output.iterdir()), {self.csv, self.chart})

    def test_failed_rollback_preserves_original_csv_for_recovery(self):
        self.csv.write_bytes(b"old predictions")
        self.chart.write_bytes(b"old figure")
        replace = Path.replace

        def interrupted(path, target):
            if target == self.chart or path.name == "previous.csv":
                raise PermissionError("replacement blocked")
            return replace(path, target)

        with patch.object(Path, "replace", interrupted), self.assertRaisesRegex(OSError, "Recovery files"):
            self.export()
        self.assertEqual((self.stage / "previous.csv").read_bytes(), b"old predictions")
        self.assertEqual(self.chart.read_bytes(), b"old figure")
        with self.assertRaises(FileExistsError):
            self.export()

    def test_another_writer_keeps_ownership_of_its_staging_directory(self):
        self.stage.mkdir()
        outstanding = self.stage / "previous.csv"
        outstanding.write_bytes(b"another export")
        with self.assertRaisesRegex(FileExistsError, "active or interrupted"):
            self.export()
        self.assertEqual(outstanding.read_bytes(), b"another export")
        self.assertFalse(self.csv.exists())
        self.assertFalse(self.chart.exists())
