import ast
import json
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "benchmark" / "benchmark_maxim_baseline_shared.ipynb"


def load_notebook():
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def code_source(notebook):
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


class BenchmarkNotebookTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = load_notebook()
        cls.source = code_source(cls.notebook)

    def test_notebook_is_clean_and_every_code_cell_parses(self):
        self.assertEqual(self.notebook["nbformat"], 4)
        for index, cell in enumerate(self.notebook["cells"]):
            if cell.get("cell_type") != "code":
                continue
            self.assertIsNone(cell.get("execution_count"))
            self.assertEqual(cell.get("outputs"), [])
            ast.parse("".join(cell.get("source", [])), filename=f"cell_{index}")

    def test_uses_the_fixed_checkpoints_without_fallback(self):
        self.assertIn("baseline_S-2_from_newsplit_ckpt_extra10", self.source)
        self.assertIn("moe_shared_residual_from_newsplit_baseline_extra10", self.source)
        self.assertIn('"expected_step": 20000', self.source)
        self.assertIn('"expected_step": 16000', self.source)
        self.assertIn("validate_tree(template, variables", self.source)
        self.assertIn("no se restauró un TrainState completo", self.source)
        self.assertNotIn("fallback", self.source.lower().split("sin fallback aleatorio", 1)[-1])

    def test_benchmark_is_synchronized_and_separates_compilation(self):
        self.assertIn("block_until_ready", self.source)
        self.assertIn("WARMUP_EXECUTIONS = 10", self.source)
        self.assertIn("TIMED_EXECUTIONS = 100", self.source)
        self.assertIn("TIMING_BLOCKS = 3", self.source)
        self.assertIn("jitted_function.lower(*args)", self.source)
        self.assertIn("compilation_seconds", self.source)
        self.assertIn("time.perf_counter_ns()", self.source)

    def test_collects_requested_cost_dimensions(self):
        for marker in (
            "cost_analysis()",
            "memory_analysis()",
            "NvmlProcessSampler",
            "total_parameters",
            "trainable_parameters",
            "executed_current_parameters",
            "median_ms_per_step",
            "median_ms_per_image",
        ):
            self.assertIn(marker, self.source)

    def test_compares_current_and_equivalent_inference_routes(self):
        self.assertIn('for output_mode in ("full", "final")', self.source)
        self.assertIn("MEASURE_SPECIALIZED_SHARED = True", self.source)
        self.assertIn("dynamic_static_equivalence_max_abs_diff", self.source)
        self.assertIn("full_final_equivalence_max_abs_diff", self.source)
        self.assertIn("maximum >= 1e-6", self.source)

    def test_training_benchmark_is_non_persistent_but_has_real_update(self):
        self.assertIn("jax.value_and_grad", self.source)
        self.assertIn("state.apply_gradients", self.source)
        self.assertIn("optax.adamw", self.source)
        self.assertIn("El paso de optimizador no avanzó exactamente uno", self.source)
        self.assertNotIn("checkpoints.save_checkpoint", self.source)

    def test_outputs_are_reanudable_and_do_not_claim_results(self):
        self.assertIn("benchmark_results.json", self.source)
        self.assertIn("diagnostico_costo_maxim.md", self.source)
        self.assertIn("measurement_complete", self.source)
        self.assertIn("atomic_json", self.source)
        self.assertIn("No implementar S1/S2", self.source)

    def test_restarts_preserve_comparable_environment_and_requested_order(self):
        self.assertIn("comparable_environment", self.source)
        self.assertIn(
            'La corrida reanudada usa otra GPU, driver o entorno', self.source
        )
        self.assertIn(
            'selected_specs = [specs_by_name[name] for name in MODEL_FILTER]',
            self.source,
        )

    def test_colab_profiler_failure_is_recoverable(self):
        self.assertIn("RUN_PROFILING = False", self.source)
        self.assertIn("PLUGIN_Profiler_Api", self.source)
        self.assertIn('previous_execution["status"] = "interrupted"', self.source)
        self.assertIn("Omitiendo resolución completa", self.source)
        self.assertIn("profile_pending", self.source)


if __name__ == "__main__":
    unittest.main()
