"""
A program that demonstrates templates knowledge graph generation from log files
as well as knowledge graph completion using both traditional and GNN models
"""

import os

from kg_generation import extract_dataset
from kg_generation import kgg
from kg_completion import kgc


def parent_dir(path: str) -> str:
    """
    Return the path to the parent directory of a file

    Parameters
    ----------
    - `path`: path to file
    """
    return path[:path.rfind(os.sep)]

def main():
    """
    The demo code
    """
    raw_data_dir = os.path.join(parent_dir(os.path.dirname(__file__)), "data", "AIT-LDS-v1_1")
    labels = True
    exclude_errors = True
    extract_dataset.extract_dataset(raw_data_dir, exclude_errors)
    kgg.gen_kg(raw_data_dir, labels)

    preprocessed_data_dir = os.path.join(raw_data_dir, "preprocessed")
    train_path = os.path.join(preprocessed_data_dir, "train.ttl")
    test_path = os.path.join(preprocessed_data_dir, "test.ttl")
    val_path = os.path.join(preprocessed_data_dir, "val.ttl")
    results_path = "results"
    kgc.kgc(train_path, test_path, val_path, results_path, "AIT")

if __name__ == "__main__":
    main()
