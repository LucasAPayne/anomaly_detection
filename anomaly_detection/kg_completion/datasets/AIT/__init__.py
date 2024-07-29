# -*- coding: utf-8 -*-

"""Get triples from the AIT dataset."""

import json
import pathlib
from docdata import parse_docdata
from pykeen.datasets.base import PathDataset

__all__ = [
    "AIT_TRAIN_PATH",
    "AIT_TEST_PATH",
    "AIT_VALIDATE_PATH",
    "AIT",
]

HERE = pathlib.Path(__file__).resolve().parent

AIT_TRAIN_PATH = HERE.joinpath("train.txt")
AIT_TEST_PATH = HERE.joinpath("test.txt")
AIT_VALIDATE_PATH = HERE.joinpath("valid.txt")

@parse_docdata
class AIT(PathDataset):
    """The AIT dataset."""

    def __init__(self, **kwargs):
        """Initialize the AIT dataset.

        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.PathDataset`.
        """
        super().__init__(
            training_path=AIT_TRAIN_PATH,
            testing_path=AIT_TEST_PATH,
            validation_path=AIT_VALIDATE_PATH,
            create_inverse_triples=True,
            **kwargs,
        )
        self._load_metadata()

    def _load_metadata(self):
        """Load labels for downstream classification task."""

        test_labels = []
        test_log_ids = []
        with open(AIT_TEST_PATH, "r", encoding="utf-8") as test_file:
            for line in test_file:
                values = line.rstrip().split('\t')
                label = values[3]
                log_id = values[4]
                test_labels.append(label)
                test_log_ids.append(log_id)

        val_labels = []
        val_log_ids = []
        with open(AIT_VALIDATE_PATH, "r", encoding="utf-8") as val_file:
            for line in val_file:
                values = line.rstrip().split('\t')
                label = values[3]
                log_id = values[4]
                val_labels.append(label)
                val_log_ids.append(log_id)

        self.metadata = {"test": {"labels": test_labels, "log_ids": test_log_ids},
                         "valid": {"labels": val_labels, "log_ids": val_log_ids}}

        with open(HERE.joinpath("metadata.txt"), "w", encoding="utf-8") as meta_file:
            json.dump(self.metadata["test"], meta_file, indent=4)

if __name__ == "__main__":
    AIT().summarize()
