import ast
import hashlib
import json
import re
from pathlib import Path
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "benchmark" / "benchmark_maxim_v2_compresion.ipynb"


def load_notebook():
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def code_source(notebook):
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


class BenchmarkV2NotebookTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = load_notebook()
        cls.source = code_source(cls.notebook)

    def test_clean_cells_that_parse(self):
        self.assertEqual(self.notebook["nbformat"], 4)
        for index, cell in enumerate(self.notebook["cells"]):
            if cell.get("cell_type") != "code":
                continue
            self.assertIsNone(cell.get("execution_count"))
            self.assertEqual(cell.get("outputs"), [])
            ast.parse("".join(cell["source"]), filename=f"cell_{index}")

    def test_pinned_source_hashes_match_repository(self):
        match = re.search(
            r"EXPECTED_SOURCE_SHA256 = (\{.*?\})\n", self.source, re.S)
        self.assertIsNotNone(match)
        expected = json.loads(match.group(1))
        self.assertEqual(set(expected), {
            "maxim/models/maxim.py", "maxim/models/compression.py",
            "maxim/models/compat.py"})
        for relative, digest in expected.items():
            actual = hashlib.sha256((REPO_ROOT / relative).read_bytes())
            self.assertEqual(actual.hexdigest(), digest, relative)

    def test_fixed_checkpoints_and_strict_restore(self):
        for marker in (
            "baseline_S-2_from_newsplit_ckpt_extra10",
            "moe_shared_residual_from_newsplit_baseline_extra10",
            '"expected_step": 20000', '"expected_step": 16000',
            "validate_tree(template_variables", "transplant_params(",
            "no se restauró un TrainState completo",
        ):
            self.assertIn(marker, self.source)

    def test_same_process_interleaved_protocol(self):
        for marker in (
            "def run_interleaved", "order[::-1]", "block_until_ready",
            "WARMUP_EXECUTIONS = 10", "TIMED_EXECUTIONS = 100",
            "TIMING_BLOCKS = 3", "jax_compilation_cache_dir",
            "lowered.cost_analysis", "kernel_launch_estimate",
            "cudnn_algorithms",
        ):
            self.assertIn(marker, self.source)

    def test_requested_variants_and_checks(self):
        for variant in (
            '"B0"', '"S0"', '"S0_sel"', '"S1_d32"', '"S2_d32"',
            '"S2_d32_todos"', '"S1_d64"', '"S2_d64"', '"S0_sel_sin_rcab2"',
            '"ref_sin_cross_gating"', '"ref_sin_gmlp"', '"B0@bf16"',
            '("B0_remat", "full")', '("S2_d32", "new_only")',
        ):
            self.assertIn(variant, self.source)
        for marker in (
            "selección efectiva de la cabeza", "selección efectiva del banco",
            "S2 ejecuta tantas convoluciones como S1",
            "rematerialización conserva la pérdida", "STRICT_CHECKS = True",
        ):
            self.assertIn(marker, self.source)

    def test_no_training_persistence_or_quality_claims(self):
        self.assertNotIn("save_checkpoint", self.source)
        self.assertIn("Sólo costo", self.source)
        self.assertIn("benchmark_v2_results.json", self.source)
        self.assertIn("diagnostico_v2_compresion.md", self.source)


if __name__ == "__main__":
    unittest.main()
