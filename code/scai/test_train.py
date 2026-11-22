import tempfile
import unittest
from pathlib import Path

import torch

from train import StratocumulusDataset, evaluate_counts


class EvaluationTests(unittest.TestCase):
    def test_only_class_folders_identify_positives(self):
        with tempfile.TemporaryDirectory(prefix="stratocumulus") as root:
            for folder in ["Cu", "Sc", "stratocumulus clouds"]:
                directory = Path(root, folder)
                directory.mkdir()
                (directory / "DSC_123.jpg").touch()
            dataset = StratocumulusDataset(root)
            self.assertEqual(dataset.labels, [0, 1, 1])

    def test_always_positive_across_uneven_batches(self):
        batches = [(torch.ones(2, 1), torch.tensor([0, 1])),
                   (torch.ones(1, 1), torch.tensor([0]))]
        counts = evaluate_counts(torch.nn.Identity(), batches, "cpu")
        self.assertEqual(counts, (1, 0, 2, 0))
