import unittest

from scripts.create_grouped_splits import classify, create_task_split


class FilenameGroupingTest(unittest.TestCase):
    def test_known_filename_families(self):
        cases = {
            ("deblur", "GOPR0374_11_02-000572.png"): "gopro-sequence:GOPR0374_11_02",
            ("deblur", "166fromGOPR0955.png"): "gopro-video:GOPR0955",
            ("deblur", "RB_scene228_blur_10.png"): "realblur:rb_scene228",
            ("dehaze", "SOTS_indoor_1400_10.png"): "sots_indoor_1400",
            ("dehaze", "SOTS_outdoor_1981_0.8_0.2.jpg"): "sots_outdoor_1981",
            ("denoise", "SIDD_0062_003_S6_03200_02500_4400_L_001.png"): "sidd:0062",
            ("derain", "rain_light_train-34x2.png"): "rain:train:34",
            ("enhance", "a3291-LS051026_day_2_arive38.png"): "fivek:3291",
        }
        for (task, name), expected in cases.items():
            with self.subTest(task=task, name=name):
                self.assertEqual(classify(task, name)[0], expected)


class GroupedSplitTest(unittest.TestCase):
    def test_existing_test_group_is_closed_before_selecting_validation(self):
        train = [
            "SOTS_indoor_1400_1.png",
            "SOTS_indoor_1400_2.png",
            "SOTS_indoor_1401_1.png",
            "SOTS_indoor_1402_1.png",
        ]
        test = ["SOTS_indoor_1400_3.png"]
        splits, metadata = create_task_split("dehaze", train, test, 0.5, 42)

        self.assertEqual(
            splits["test"],
            ["SOTS_indoor_1400_3.png", "SOTS_indoor_1400_1.png", "SOTS_indoor_1400_2.png"],
        )
        self.assertEqual(metadata["historical_train_groups_moved_to_test"], 1)
        self.assertEqual(
            metadata["historical_train_group_ids_moved_to_test"],
            ["sots_indoor_1400"],
        )
        groups = {
            split: {classify("dehaze", name)[0] for name in names}
            for split, names in splits.items()
        }
        self.assertFalse(groups["train"] & groups["val"])
        self.assertFalse(groups["train"] & groups["test"])
        self.assertFalse(groups["val"] & groups["test"])


if __name__ == "__main__":
    unittest.main()
