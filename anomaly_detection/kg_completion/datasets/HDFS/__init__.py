# -*- coding: utf-8 -*-

"""Get triples from the HDFS dataset."""

import json
import pathlib
from docdata import parse_docdata
from pykeen.datasets.base import PathDataset

__all__ = [
    "HDFS_TRAIN_PATH",
    "HDFS_TEST_PATH",
    "HDFS_VALIDATE_PATH",
    "HDFS",
]

HERE = pathlib.Path(__file__).resolve().parent

HDFS_TRAIN_PATH = HERE.joinpath("train.txt")
HDFS_TEST_PATH = HERE.joinpath("test.txt")
HDFS_VALIDATE_PATH = HERE.joinpath("valid.txt")

# NOTE(lucas): When a TriplesFactory is made, np.unique gets called on them,
# which changes the order of the triples. This is undesirable for creating metadata,
# so that statement should be removed.

@parse_docdata
class HDFS(PathDataset):
    """The HDFS dataset."""

    def __init__(self, **kwargs):
        """Initialize the HDFS dataset.

        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.PathDataset`.
        """
        super().__init__(
            training_path=HDFS_TRAIN_PATH,
            testing_path=HDFS_TEST_PATH,
            validation_path=HDFS_VALIDATE_PATH,
            create_inverse_triples=True,
            **kwargs,
        )
        self._load_metadata()

    def _load_metadata(self):
        """Load labels for downstream classification task."""

        test_labels = []
        test_log_ids = []
        with open(HDFS_TEST_PATH, "r", encoding="utf-8") as test_file:
            for line in test_file:
                values = line.rstrip().split('\t')
                label = values[3]
                log_id = values[4]
                test_labels.append(label)
                test_log_ids.append(log_id)

        val_labels = []
        val_log_ids = []
        with open(HDFS_VALIDATE_PATH, "r", encoding="utf-8") as val_file:
            for line in val_file:
                values = line.rstrip().split('\t')
                label = values[3]
                log_id = values[4]
                val_labels.append(label)
                val_log_ids.append(log_id)

        self.metadata = {"test": {"labels": test_labels, "log_ids": test_log_ids},
                         "valid": {"labels": val_labels, "log_ids": val_log_ids}}

        with open(HERE.joinpath("metadata.json"), "w", encoding="utf-8") as meta_file:
            json.dump(self.metadata["test"], meta_file, indent=4)

if __name__ == "__main__":
    HDFS().summarize()
