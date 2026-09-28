import json
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "maxim" / "moeTrainer.ipynb"


def notebook_source():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


class SharedWarmStartNotebookTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = notebook_source()

    def test_uses_selected_baseline_and_additional_budget(self):
        self.assertIn('EXTRA_EPOCHS = 10', self.source)
        self.assertIn('NUM_EPOCHS = EXTRA_EPOCHS', self.source)
        self.assertIn('BASELINE_CHECKPOINT_DIR = os.path.join(', self.source)
        self.assertIn('moe_shared_residual_from_baseline_extra10', self.source)
        self.assertIn('RESUME_MODE = "none"', self.source)

    def test_requires_exact_parameter_coverage(self):
        self.assertIn('if source_keys != target_keys:', self.source)
        self.assertIn('if shape_mismatches:', self.source)
        self.assertNotIn('loaded > 0.9 * backbone_total', self.source)
        self.assertIn('parameter_tree_sha256', self.source)

    def test_zero_residuals_and_equivalence_are_mandatory(self):
        self.assertIn('zero_init_output=True', self.source)
        self.assertIn('output kernel is not zero', self.source)
        self.assertIn('equivalence_max_abs_diff >= 1e-6', self.source)

    def test_optimizer_is_reset_and_initial_checkpoint_is_preserved(self):
        self.assertIn('assert int(state.step) == 0', self.source)
        self.assertIn('INITIAL_CHECKPOINT_DIR', self.source)
        self.assertIn('initial_metrics.json', self.source)
        self.assertIn('best_epoch = 0', self.source)
        self.assertIn('"additional_steps": EXTRA_EPOCHS * steps_per_epoch', self.source)

    def test_documents_historical_split_limitation(self):
        self.assertIn('predates the new', self.source)
        self.assertIn('not as the final blind-test comparison', self.source)


if __name__ == "__main__":
    unittest.main()
