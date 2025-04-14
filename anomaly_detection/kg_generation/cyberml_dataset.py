"""
Script that extracts the cyberML dataset, which is made up logs already preprocessed into a KG.
The training set contains normal activity.
The testing set contains both normal and malicious activity.
"""

import os

from . import kg_generation

def extract_train_set(root_dir: str, out_path: str) -> None:
    """
    Extract train set. The train set is already preprocessed into a KG and is an appropriate format,
    so just copy the file.

    Parameters
    ----------
    - `root_dir`: root directory of cyberML dataset
    - `preprocessed_data_dir`: directory to output preprocessed files
    """
    in_train_file = os.path.join(root_dir, "training", "train.txt")
    buffer = []
    lines = []
    with open(in_train_file, "r", encoding="utf-8") as infile:
        lines = infile.readlines()

    for line in lines:
        buffer.append(line.rstrip() + '\t0\n')

    with open(out_path, "w", encoding="utf-8") as outfile:
        outfile.writelines(buffer)

def extract_test_set(root_dir: str, out_path: str) -> None:
    """
    Extract test set. Test files are divided into categories, so concatenate all data into one file.

    Parameters
    ----------
    - `root_dir`: root directory of cyberML dataset
    - `preprocessed_data_dir`: directory to output preprocessed files
    """
    test_dir = os.path.join(root_dir, "test")
    in_test_files = ["credential_use.txt", "https.txt", "scan.txt",
                    "ssh.txt", "variables_access.txt"]

    buffer = []
    # Loop through test file and write contents to one large test file
    for file in in_test_files:
        test_file = os.path.join(test_dir, file)
        lines = []
        with open(test_file, "r", encoding="utf-8") as infile:
            lines = infile.readlines()

        for line in lines:
            split_line = line.split('\t')[:-1]
            out_line = '\t'.join(split_line)

            # NOTE(lucas): For now, labels will changed for binary classifier
            # [0, 1, 2] -> 1: suspicious
            # [3,4] -> 0: normal
            label = int(line.split('\t')[-1].strip())
            if label in [0, 1, 2]:
                label = 1
            elif label in [3, 4]:
                label = 0

            buffer.append(out_line + '\t' + str(label) + '\n')

    with open(out_path, "w", encoding="utf-8") as out_test_file:
        out_test_file.writelines(buffer)

def extract_dataset(root_dir: str, val_ratio: float) -> None:
    """
    Extract data from the cyberML dataset.
    Use a portion based on `val_ratio` of the training dataset for validation data.
    Note that this dataset has already been preprocessed into a knowledge graph.

    Parameters
    ----------
    - `root_dir`: root directory of CyberML dataset
    - `val_ratio`: ratio of test data to be used as validation data
    """
    preprocessed_data_dir = os.path.join(root_dir, "preprocessed")
    if not os.path.exists(preprocessed_data_dir):
        os.mkdir(preprocessed_data_dir)

    kgc_prefix = kg_generation.join_path("..", "kg_completion", "datasets", "CyberML")
    train_path = kg_generation.join_path(kgc_prefix, "train.txt")
    test_path = kg_generation.join_path(kgc_prefix, "test.txt")
    val_path = kg_generation.join_path(kgc_prefix, "valid.txt")

    extract_train_set(root_dir, train_path)
    extract_test_set(root_dir, test_path)
    kg_generation.generate_val_set(test_path, val_path, val_ratio)
