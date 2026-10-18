import json
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    REPO_ROOT / "notebooks" / "entrenamiento" / "maximTrainer_no_moe.ipynb",
    REPO_ROOT / "notebooks" / "entrenamiento" / "moeTrainer.ipynb",
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

    def test_baseline_validation_adds_ssim_without_changing_selection(self):
        source = notebook_source(REPO_ROOT / "notebooks" / "entrenamiento" / "maximTrainer_no_moe.ipynb")

        self.assertIn("def compute_ssim_per_example", source)
        self.assertIn('"sample_ssim": compute_ssim_per_example', source)
        self.assertIn("aggregate_task_ssim(task_ssim_sum, task_count)", source)
        self.assertIn('"ssim_weighted"', source)
        self.assertIn('"ssim_macro"', source)
        self.assertIn('"task_ssim"', source)
        self.assertIn("tf.image.ssim(", source)
        self.assertIn("CHECKPOINT_SELECTION_METRIC = SELECTION_METRIC", source)

        train_step_source = source.split("def train_step", 1)[1].split(
            "def eval_step", 1
        )[0]
        self.assertNotIn("compute_ssim_per_example", train_step_source)

    def test_new_baseline_run_is_fresh_and_uses_a_new_output_name(self):
        source = notebook_source(REPO_ROOT / "notebooks" / "entrenamiento" / "maximTrainer_no_moe.ipynb")

        self.assertIn('RUN_NAME = f"baseline_{ACTIVE_VARIANT}_newsplit_{init_tag}"', source)
        self.assertIn('RESUME_MODE = "none"', source)
        self.assertIn('MULTITASK_INIT_CKPT = ""', source)
        self.assertIn('"split_manifest": SPLIT_MANIFEST', source)
        self.assertNotIn("smoke_test_drive_splits.py", source)
        self.assertIn("SSIM_DATA_RANGE = 1.0", source)
        self.assertIn("SSIM_FILTER_SIZE = 11", source)
        self.assertIn("SSIM_FILTER_SIGMA = 1.5", source)
        self.assertIn("SSIM_K1 = 0.01", source)
        self.assertIn("SSIM_K2 = 0.03", source)

        notebook = json.loads(
            (REPO_ROOT / "notebooks" / "entrenamiento" / "maximTrainer_no_moe.ipynb").read_text(
                encoding="utf-8"
            )
        )
        runtime_config = "".join(notebook["cells"][5].get("source", []))
        auxiliary_functions = "".join(notebook["cells"][7].get("source", []))
        self.assertNotIn("def build_split_manifest", runtime_config)
        self.assertIn("def build_split_manifest", auxiliary_functions)


if __name__ == "__main__":
    unittest.main()
