import csv
import tempfile
import unittest
from pathlib import Path

from evaluation.protocol import load_lfw_pair_lists, save_lfw_pair_lists


class LfwProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.data_dir = self.root / "metadata"
        self.image_root = self.root / "images"
        self.data_dir.mkdir()

        self._write_csv("matchpairsDevTrain.csv", [
            ["name", "imagenum1", "imagenum2"],
            ["TrainPerson", "1", "2"],
        ])
        self._write_csv("mismatchpairsDevTrain.csv", [
            ["name", "imagenum1", "name", "imagenum2"],
            ["TrainPerson", "1", "TrainOther", "1"],
        ])
        self._write_csv("matchpairsDevTest.csv", [
            ["name", "imagenum1", "imagenum2"],
            ["TestPerson", "1", "2"],
        ])
        self._write_csv("mismatchpairsDevTest.csv", [
            ["name", "imagenum1", "name", "imagenum2"],
            ["TestPerson", "1", "TestOther", "1"],
        ])
        for identity, numbers in {
            "TrainPerson": (1, 2), "TrainOther": (1,),
            "TestPerson": (1, 2), "TestOther": (1,),
        }.items():
            identity_dir = self.image_root / identity
            identity_dir.mkdir(parents=True)
            for number in numbers:
                (identity_dir / f"{identity}_{number:04d}.jpg").touch()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _write_csv(self, filename, rows):
        with (self.data_dir / filename).open("w", newline="", encoding="utf-8") as output:
            csv.writer(output).writerows(rows)

    def test_loads_same_and_different_pairs_with_disjoint_identities(self):
        pair_lists = load_lfw_pair_lists(self.data_dir, self.image_root)

        self.assertEqual([pair.label for pair in pair_lists["validation"]], [1, 0])
        self.assertEqual([pair.label for pair in pair_lists["test"]], [1, 0])
        validation_ids = {
            identity for pair in pair_lists["validation"]
            for identity in (pair.identity_a, pair.identity_b)
        }
        test_ids = {
            identity for pair in pair_lists["test"]
            for identity in (pair.identity_a, pair.identity_b)
        }
        self.assertFalse(validation_ids & test_ids)
        self.assertTrue(Path(pair_lists["validation"][0].image_a).is_file())

    def test_rejects_identity_overlap(self):
        self._write_csv("matchpairsDevTest.csv", [
            ["name", "imagenum1", "imagenum2"],
            ["TrainPerson", "1", "2"],
        ])
        self._write_csv("mismatchpairsDevTest.csv", [
            ["name", "imagenum1", "name", "imagenum2"],
            ["TrainPerson", "1", "TrainOther", "1"],
        ])

        with self.assertRaisesRegex(ValueError, "identities overlap"):
            load_lfw_pair_lists(self.data_dir, self.image_root)

    def test_pair_lists_refuse_to_overwrite(self):
        pair_lists = load_lfw_pair_lists(self.data_dir, self.image_root)
        output_dir = self.root / "results"
        save_lfw_pair_lists(pair_lists, output_dir)

        with self.assertRaises(FileExistsError):
            save_lfw_pair_lists(pair_lists, output_dir)


if __name__ == "__main__":
    unittest.main()