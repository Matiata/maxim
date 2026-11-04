import ast
import hashlib
import json
import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ADAPTATION = REPO_ROOT / "notebooks" / "entrenamiento" / "adaptacion_compresion.ipynb"
ADAPTATION_SEED2 = (REPO_ROOT / "notebooks" / "entrenamiento"
                    / "adaptacion_compresion_semilla2.ipynb")
ADAPTATION_L012 = (REPO_ROOT / "notebooks" / "entrenamiento"
                   / "adaptacion_compresion_todos_niveles.ipynb")
ADAPTATION_L12 = (REPO_ROOT / "notebooks" / "entrenamiento"
                  / "adaptacion_compresion_niveles_1_2.ipynb")
VALIDATION = REPO_ROOT / "notebooks" / "evaluacion" / "validacion_compresion.ipynb"
VALIDATION_LEVELS = (REPO_ROOT / "notebooks" / "evaluacion"
                     / "validacion_compresion_niveles.ipynb")
TEST = REPO_ROOT / "notebooks" / "evaluacion" / "test_final_compresion.ipynb"
TEST_LEVELS = (REPO_ROOT / "notebooks" / "evaluacion"
               / "test_final_compresion_niveles.ipynb")
BENCHMARK_V2 = (REPO_ROOT / "notebooks" / "benchmark"
                / "benchmark_maxim_v2_compresion.ipynb")
BENCHMARK_LEVELS = (REPO_ROOT / "notebooks" / "benchmark"
                    / "benchmark_maxim_niveles.ipynb")
ALL = (ADAPTATION, ADAPTATION_SEED2, ADAPTATION_L012, ADAPTATION_L12,
       VALIDATION, VALIDATION_LEVELS, TEST, TEST_LEVELS)


def code_source(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return notebook, "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"]
        if cell["cell_type"] == "code")


class CompressionNotebooksTest(unittest.TestCase):

    def test_cells_are_clean_and_parse(self):
        for path in ALL:
            notebook, _ = code_source(path)
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] != "code":
                    continue
                self.assertIsNone(cell.get("execution_count"), path.name)
                self.assertEqual(cell.get("outputs"), [], path.name)
                ast.parse("".join(cell["source"]), filename=f"{path.name}:{index}")

    def test_pinned_model_sources_match_repository(self):
        for path in ALL:
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
            'SPLIT = "val"', 'REPLICAS = ["v1", "v2"]',
            "same_protocol_within_replica",
            "same_protocol_except_seeds_across_replicas",
            "distinct_seeds_per_replica",
            "adaptation_teacher_is_shared_checkpoint", "def stratified_bootstrap",
            "SEED_AVERAGED", "REFERENCE_TOLERANCE_DB",
            "REQUIRE_COMPLETE_ADAPTATION = True", "compresion_validacion_semillas",
        ):
            self.assertIn(marker, source)
        self.assertNotIn("save_checkpoint", source)

    def test_levels_validation_protocol(self):
        _, source = code_source(VALIDATION_LEVELS)
        for marker in (
            'SPLIT = "val"', 'REPLICAS = ["v1", "v2"]',
            '("l12", (1, 2)), ("l012", (0, 1, 2))', "VARIANTS.update(LEVEL_VARIANTS)",
            '"v2": ["S1_d32", "S2_d32", "S2_d32_l12"]',
            '"S2_d32_l12 - S0_sel": [(f"S2_d32_l12_{r}", "S0_sel")',
            '"S2_d32_l12 - S2_d32": [(f"S2_d32_l12_{r}", f"S2_d32_{r}")',
            "same_protocol_except_seeds_across_replicas", "variant_spec_matches",
            "_reproduces_selection_db", "SELECTION_TOLERANCE_DB = 0.01",
            "CRITERION_MACRO_DB = 0.05", "CRITERION_TASK_DB = 0.10",
            'with jax.default_device(jax.devices("cpu")[0]):',
            "REQUIRE_COMPLETE_ADAPTATION = True", "compresion_validacion_niveles",
        ):
            self.assertIn(marker, source)
        self.assertNotIn("save_checkpoint", source)
        self.assertNotIn("compresion_validacion_semillas", source)

    def test_replica_seeds_keep_original_protocol(self):
        # v1 must keep the seeds of the runs already completed (protocol hash).
        _, source = code_source(ADAPTATION)
        self.assertIn('"v1": {"seed": 42, "new_module_seed": 1234}', source)
        self.assertIn('REPLICA = "v1"', source)
        self.assertIn("RUN_TAG = REPLICA", source)

    def test_seed2_notebook_only_changes_replica_default(self):
        first, _ = code_source(ADAPTATION)
        second, _ = code_source(ADAPTATION_SEED2)
        a = ["".join(c["source"]) for c in first["cells"] if c["cell_type"] == "code"]
        b = ["".join(c["source"]) for c in second["cells"] if c["cell_type"] == "code"]
        self.assertEqual(len(a), len(b))
        different = [(x, y) for x, y in zip(a, b) if x != y]
        self.assertEqual(len(different), 1)
        x, y = different[0]
        self.assertEqual(x.replace('REPLICA = "v1"', 'REPLICA = "v2"'), y)

    def assert_only_replaced_levels_change(self, path, markers):
        # Mismo protocolo que S1/S2_d32; sólo cambian los niveles sustituidos.
        first, _ = code_source(ADAPTATION)
        second, source = code_source(path)
        a = ["".join(c["source"]) for c in first["cells"] if c["cell_type"] == "code"]
        b = ["".join(c["source"]) for c in second["cells"] if c["cell_type"] == "code"]
        self.assertEqual(len(b), len(a) + 1)
        self.assertTrue(b[-1].startswith("# 11. Comparación"))
        changed = [x.split("\n")[0] for x, y in zip(a, b) if x != y]
        self.assertEqual(changed, [
            "# 3. Configuración: elegir la variante; el resto es el protocolo común",
            "# 8. Crear o reanudar la corrida"])
        protocol = lambda text: text[text.index("TASKS = ["):
                                     text.index("if VARIANT not in")]
        config = next(x for x in b if x.startswith("# 3. Configuración"))
        self.assertEqual(protocol(a[2]), protocol(config))
        for marker in ('"rcab_mixer_width": 32',
                       '"variant_spec": json_safe(VARIANTS[VARIANT])',
                       "CRITERION_MACRO_DB = 0.05", "CRITERION_TASK_DB = 0.10",
                       *markers):
            self.assertIn(marker, source)
        self.assertNotIn('"test")', source.replace(
            'for split in ("train", "val", "test"):', ""))

    def test_all_levels_notebook_only_changes_replaced_levels(self):
        self.assert_only_replaced_levels_change(ADAPTATION_L012, (
            'VARIANT = "S2_d32_l012"', '"rcab_mixer_levels": (0, 1, 2)',
            "VARIANTS.update(OPTION1_VARIANTS)"))

    def test_levels_1_2_notebook_only_changes_replaced_levels(self):
        self.assert_only_replaced_levels_change(ADAPTATION_L12, (
            'VARIANT = "S2_d32_l12"', '("l12", (1, 2)), ("l012", (0, 1, 2))',
            'TRAINABLE_VARIANTS = ("S2_d32_l12", "S1_d32_l12")',
            "VARIANTS.update(LEVEL_VARIANTS)",
            'with jax.default_device(jax.devices("cpu")[0]):'))

    def test_final_test_notebook(self):
        _, source = code_source(TEST)
        for marker in (
            'SPLIT = "test"', 'CONFIRM_TEST_RUN = False',
            "Poner CONFIRM_TEST_RUN = True",
            '"B0": 28.12742717709229, "S0_sel": 28.13664076320675',
            "compresion_test_final", '"primary_analysis"',
        ):
            self.assertIn(marker, source)
        self.assertNotIn("save_checkpoint", source)
        self.assertNotIn("BF16_MODELS = [\"", source)

    def test_cache_does_not_retain_full_images(self):
        # Un recorte de NumPy es una vista: sin copiar, cada crop retenía la
        # imagen completa y el caché de test agotó la RAM de Colab.
        cells = []
        for path in ALL:
            notebook, _ = code_source(path)
            cells.append(next(
                "".join(c["source"]) for c in notebook["cells"]
                if "".join(c["source"]).startswith("# Validación en memoria")))
            self.assertIn("left:left + crop_size].copy()", cells[-1], path.name)
            self.assertIn("ThreadPoolExecutor(max_workers=LOAD_WORKERS)", cells[-1])
            self.assertIn(
                'task_path = cache_path.with_name(f"{cache_path.stem}_{task}.npz")',
                cells[-1])
        self.assertEqual(len(set(cells)), 1)

    def test_levels_test_notebook(self):
        _, source = code_source(TEST_LEVELS)
        for marker in (
            'SPLIT = "test"', "CONFIRM_TEST_RUN = False",
            "Poner CONFIRM_TEST_RUN = True", '"decision": (',
            '"B0": 28.12741289151419, "S0_sel": 28.136597748472962',
            '"S1_d32_v2": 28.08089575271877, "S2_d32_v2": 28.153763404056907',
            '"S2_d32_l12 - S0_sel": [(f"S2_d32_l12_{r}", "S0_sel")',
            '"S2_d32_l12 - S2_d32": [(f"S2_d32_l12_{r}", f"S2_d32_{r}")',
            "variant_spec_matches", "REQUIRE_COMPLETE_ADAPTATION = True",
            'CHECKPOINT_ROOT / "compresion_test_niveles"',
            'with jax.default_device(jax.devices("cpu")[0]):',
        ):
            self.assertIn(marker, source)
        self.assertNotIn("save_checkpoint", source)
        self.assertNotIn("_reproduces_selection_db", source)
        self.assertNotIn('CHECKPOINT_ROOT / "compresion_test_final"', source)

    def test_levels_test_only_changes_split_and_report(self):
        # Mismos modelos y comparaciones que la validación de niveles.
        first, _ = code_source(VALIDATION_LEVELS)
        second, _ = code_source(TEST_LEVELS)
        a = ["".join(c["source"]) for c in first["cells"] if c["cell_type"] == "code"]
        b = ["".join(c["source"]) for c in second["cells"] if c["cell_type"] == "code"]
        self.assertEqual(len(a), len(b))
        changed = [x.split("\n")[0] for x, y in zip(a, b) if x != y]
        self.assertEqual(changed, [
            "# 3. Configuración",
            "# 5. Datos en memoria, resultado reanudable y verificación de procedencia",
            "# 8. Reporte: calidad, diferencias por réplica y promediadas, costo"])
        models = lambda text: text[text.index("ADAPTED_RUNS = {"):
                                   text.index("# Criterio de la opción 1")]
        self.assertEqual(models(a[2]), models(b[2]))

    def test_levels_benchmark_notebook(self):
        notebook, source = code_source(BENCHMARK_LEVELS)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell.get("execution_count"))
                self.assertEqual(cell.get("outputs"), [])
                ast.parse("".join(cell["source"]))
        match = re.search(r"EXPECTED_SOURCE_SHA256 = (\{.*?\})\n", source, re.S)
        for relative, digest in json.loads(match.group(1)).items():
            actual = hashlib.sha256((REPO_ROOT / relative).read_bytes())
            self.assertEqual(actual.hexdigest(), digest, relative)
        first, _ = code_source(BENCHMARK_V2)
        a = ["".join(c["source"]) for c in first["cells"] if c["cell_type"] == "code"]
        b = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
        self.assertEqual(len(a), len(b))
        changed = [x.split("\n")[0] for x, y in zip(a, b) if x != y]
        self.assertEqual(changed, [
            "# 3. Configuración visible antes de medir",
            "# 5. Protocolo y resultado reanudable",
            "# 10. Compilación, análisis de costo y cronometraje alternado",
            "# 11. Grupo de inferencia: todas las variantes de una resolución juntas",
            "# 14. Reporte legible derivado del JSON, sin decisiones"])
        for marker in (
            "VARIANTS.update(LEVEL_VARIANTS)", '"S2_d32_l12", "S1_d32_l12"',
            '("S2_d32_l12", "new_only")', '"B0", "S0_sel",',
            'CHECKPOINT_ROOT / "benchmark_maxim_niveles"',
            'with jax.default_device(jax.devices("cpu")[0]):',
            "WARMUP_EXECUTIONS = 10", "TIMED_EXECUTIONS = 100",
        ):
            self.assertIn(marker, source)

    def test_validation_and_test_share_evaluation_code(self):
        test, _ = code_source(TEST)
        shared = ("# Validación en memoria", "# 6. Evaluación por modelo",
                  "# 7. Diferencias pareadas", "# Restauración estricta")
        for path in (VALIDATION, VALIDATION_LEVELS, TEST_LEVELS):
            val, _ = code_source(path)
            for prefix in shared:
                a = next("".join(c["source"]) for c in val["cells"]
                         if "".join(c["source"]).startswith(prefix))
                b = next("".join(c["source"]) for c in test["cells"]
                         if "".join(c["source"]).startswith(prefix))
                self.assertEqual(a, b, f"{path.name}: {prefix}")


if __name__ == "__main__":
    unittest.main()
