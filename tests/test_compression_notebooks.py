import ast
import hashlib
import json
import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ADAPTATION = REPO_ROOT / "notebooks" / "entrenamiento" / "adaptacion_compresion.ipynb"
VALIDATION = REPO_ROOT / "notebooks" / "evaluacion" / "validacion_compresion.ipynb"


def code_source(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return notebook, "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"]
        if cell["cell_type"] == "code")


class CompressionNotebooksTest(unittest.TestCase):

    def test_cells_are_clean_and_parse(self):
        for path in (ADAPTATION, VALIDATION):
            notebook, _ = code_source(path)
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] != "code":
                    continue
                self.assertIsNone(cell.get("execution_count"), path.name)
                self.assertEqual(cell.get("outputs"), [], path.name)
                ast.parse("".join(cell["source"]), filename=f"{path.name}:{index}")

    def test_pinned_model_sources_match_repository(self):
        for path in (ADAPTATION, VALIDATION):
            _, source = code_source(path)
            match = re.search(r"EXPECTED_SOURCE_SHA256 = (\{.*?\})\n", source, re.S)
            self.assertIsNotNone(match, path.name)
            for relative, digest in json.loads(match.group(1)).items():
                actual = hashlib.sha256((REPO_ROOT / relative).read_bytes())
                self.assertEqual(actual.hexdigest(), digest, f"{path.name}: {relative}")

    def test_adaptation_protocol(self):
        _, source = code_source(ADAPTATION)
        for marker in (
            'VARIANT = "S2_d32"', 'build_variant("S0_sel"', '"scope": "new_only"',
            "DISTILL_OUTPUT_WEIGHT = 1.0", "FEATURE_DISTILL_WEIGHT = 0.1",
            "ZERO_INIT_NEW_OUTPUT = True", "capture_intermediates=is_replaced_rcab",
            "transplant_params(", "COMPARABLE_PROTOCOL_HASH",
            "Sólo se pueden agregar fases al final",
            "Los parámetros congelados cambiaron",
            'build_best_metric_payload(\n        0, initial_val, TASKS, "psnr_macro")',
            "SEED * 1000 + int(epoch)", "donate_argnums=(0, 2)",
            "jnp.array(a, copy=True)", "TEACHER_VAL_TOLERANCE_DB",
        ):
            self.assertIn(marker, source)

    def test_adaptation_never_evaluates_test(self):
        _, source = code_source(ADAPTATION)
        self.assertNotIn('read_split_names(task, "test")', source)
        self.assertNotIn('"test")\n', source.replace(
            'for split in ("train", "val", "test"):', ""))

    def test_validation_protocol(self):
        _, source = code_source(VALIDATION)
        for marker in (
            '"split": "val"', "same_adaptation_protocol",
            "adaptation_teacher_is_shared_checkpoint", "def paired_bootstrap",
            'BF16_MODELS = ["B0", "S0_sel", "S1_d32", "S2_d32"]',
            '("S2_d32", "S1_d32")', "REFERENCE_TOLERANCE_DB",
            "REQUIRE_COMPLETE_ADAPTATION = True",
        ):
            self.assertIn(marker, source)
        self.assertNotIn("save_checkpoint", source)


if __name__ == "__main__":
    unittest.main()
