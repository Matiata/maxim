import json
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    REPO_ROOT / "maxim" / "maximTrainer_no_moe.ipynb",
    REPO_ROOT / "maxim" / "moeTrainer.ipynb",
    REPO_ROOT / "maxim" / "moeTrainer_latent_k8.ipynb",
)


def notebook_source(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


class NotebookEvaluationProtocolTest(unittest.TestCase):
    def test_training_uses_validation_split_and_macro_selection(self):
        for path in NOTEBOOKS:
            with self.subTest(notebook=path.name):
                source = notebook_source(path)
                self.assertIn('split_name="val"', source)
                self.assertIn("CHECKPOINT_SELECTION_METRIC = SELECTION_METRIC", source)
                self.assertIn("val_metrics['psnr_macro']", source)
                self.assertNotIn("evaluate(state, test_dataset", source)

    def test_final_test_is_disabled_and_does_not_select_checkpoints(self):
        for path in NOTEBOOKS:
            with self.subTest(notebook=path.name):
                source = notebook_source(path)
                self.assertIn("RUN_FINAL_TEST = False", source)
                self.assertIn('split_name="test"', source)
                final_test_source = source.split("RUN_FINAL_TEST = False", 1)[1]
                self.assertNotIn("save_checkpoint(", final_test_source)

    def test_missing_validation_split_fails_instead_of_using_test(self):
        for path in NOTEBOOKS:
            with self.subTest(notebook=path.name):
                source = notebook_source(path)
                self.assertIn("Missing required {split_name} split", source)
                self.assertIn("the notebook will not derive it silently", source)

    def test_saved_metadata_is_explicit(self):
        for path in NOTEBOOKS:
            with self.subTest(notebook=path.name):
                source = notebook_source(path)
                self.assertIn("build_best_metric_payload(", source)
                self.assertIn('"psnr_weighted"', source)
                self.assertIn('"psnr_macro"', source)
                self.assertIn('"task_psnr"', source)
                self.assertIn('"task_count"', source)


if __name__ == "__main__":
    unittest.main()
