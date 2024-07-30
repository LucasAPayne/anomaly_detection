# -*- coding: utf-8 -*-

"""Get triples from the CyberML dataset."""

import json
import pathlib
from docdata import parse_docdata
from pykeen.datasets.base import PathDataset

__all__ = [
    "CYBERML_TRAIN_PATH",
    "CYBERML_TEST_PATH",
    "CYBERML_VALIDATE_PATH",
    "CyberML",
]

HERE = pathlib.Path(__file__).resolve().parent

CYBERML_TRAIN_PATH = HERE.joinpath("train.txt")
CYBERML_TEST_PATH = HERE.joinpath("test.txt")
CYBERML_VALIDATE_PATH = HERE.joinpath("valid.txt")

# NOTE(lucas): When a TriplesFactory is made, np.unique gets called on them,
# which changes the order of the triples. This is undesirable for creating metadata,
# so that statement should be removed.

@parse_docdata
class CyberML(PathDataset):
    """The CyberML dataset."""

    def __init__(self, **kwargs):
        """Initialize the CyberML dataset.

        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.PathDataset`.
        """
        super().__init__(
            training_path=CYBERML_TRAIN_PATH,
            testing_path=CYBERML_TEST_PATH,
            validation_path=CYBERML_VALIDATE_PATH,
            **kwargs,
        )
        self._load_metadata()

    def _load_metadata(self):
        """Load labels for downstream classification task."""

        test_labels = []
        with open(CYBERML_TEST_PATH, "r", encoding="utf-8") as test_file:
            for line in test_file:
                values = line.rstrip().split('\t')
                label = values[3]
                test_labels.append(label)

        val_labels = []
        with open(CYBERML_VALIDATE_PATH, "r", encoding="utf-8") as val_file:
            for line in val_file:
                values = line.rstrip().split('\t')
                label = values[3]
                val_labels.append(label)

        self.metadata = {"test": {"labels": test_labels, "log_ids": []},
                         "valid": {"labels": val_labels}, "log_ids": []}

        with open(HERE.joinpath("metadata.txt"), "w", encoding="utf-8") as meta_file:
            json.dump(self.metadata["test"], meta_file, indent=4)

if __name__ == "__main__":
    CyberML().summarize()
