import csv
from pathlib import Path
import tempfile
import unittest

from speech_ci.data import make_split, validate_split
from speech_ci.io import safe_identifier


class DataTests(unittest.TestCase):
    def test_fixed_partition_counts_and_disjointness(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for split, counts in (("train", (160, 120, 40)), ("test", (40, 30, 10))):
                for diagnosis, count in zip(("HC", "MCI", "Dementia"), counts):
                    for _ in range(count):
                        rows.append({"IDs": f"synthetic_{len(rows):04d}", "diagnosis": diagnosis, "Split": split})
            with (root / "meta-info.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=("IDs", "diagnosis", "Split"))
                writer.writeheader()
                writer.writerows(rows)
            first = make_split(root, root / "partition.json")
            second = make_split(root, root / "partition_copy.json")
            self.assertEqual(first, second)
            self.assertEqual([len(first[key]) for key in ("fit", "validation", "test")], [256, 64, 80])
            validate_split(first)

    def test_identifier_cannot_escape_data_directory(self):
        for invalid in ("../x", "a/b", "a\\b", "C:drive", "", "."):
            with self.assertRaises(ValueError):
                safe_identifier(invalid)

    def test_participant_overlap_is_rejected(self):
        people = [{"id": f"synthetic_{i}", "diagnosis": i} for i in range(3)]
        with self.assertRaises(ValueError):
            validate_split({"fit": people, "validation": people, "test": people})


if __name__ == "__main__":
    unittest.main()
