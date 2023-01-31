"""
Script that extracts the cyberML dataset, which is made up logs already preprocessed into a KG.
The training set contains normal activity.
The testing set contains both normal and malicious activity.
"""

import os

from . import kgg

def extract_train_set(root_dir: str, preprocessed_data_dir: str) -> None:
    """
    Extract train set. The train set is already preprocessed into a KG and is an appropriate format,
    so just copy the file.

    Parameters
    ----------
    - `root_dir`: root directory of cyberML dataset
    - `preprocessed_data_dir`: directory to output preprocessed files
    """
    in_train_file = os.path.join(root_dir, "training", "train.del")
    out_train_file = os.path.join(preprocessed_data_dir, "train.txt")

    # NOTE(lucas): Could use shutil for easier copying, but seems pointless to bring it in
    # to use once on such a trivial task
    with open(in_train_file, "r", encoding="utf-8") as in_file, \
         open(out_train_file, "w", encoding="utf-8") as out_file:
        for line in in_file:
            out_file.write(line)

def extract_test_set(root_dir: str, preprocessed_data_dir: str) -> None:
    """
    Extract test set. Test files are divided into categories, so concatenate all data into one file.
    Labels are also stripped and placed into a separate file

    Parameters
    ----------
    - `root_dir`: root directory of cyberML dataset
    - `preprocessed_data_dir`: directory to output preprocessed files
    """
    test_dir = os.path.join(root_dir, "test")
    test_files = ["credential_use.del", "https.del", "scan.del", "ssh.del", "variables_access.del"]

    # TODO(lucas): Save labels to separate file
    # Loop through test file and write contents to one large test file
    out_test_file = os.path.join(preprocessed_data_dir, "test.txt")
    with open(out_test_file, "w", encoding="utf-8") as out_file:
        for file in test_files:
            test_file = os.path.join(test_dir, file)
            with open(test_file, "r", encoding="utf-8") as in_file:
                for line in in_file:
                    split_line = line.split('\t')[:-1]
                    out_line = '\t'.join(split_line)
                    out_file.write(out_line + '\n')

def extract_dataset(root_dir: str, val_ratio: float) -> None:
    """
    Extract data from the cyberML dataset.
    Use a portion based on `val_ratio` of the training dataset for validation data.
    Note that this dataset has already been preprocessed into a knowledge graph.

    Parameters
    ----------
    - `root_dir`: root directory of CyberML dataset
    - `val_ratio`:
    """
    preprocessed_data_dir = os.path.join(root_dir, "preprocessed")
    if not os.path.exists(preprocessed_data_dir):
        os.mkdir(preprocessed_data_dir)

    extract_train_set(root_dir, preprocessed_data_dir)
    extract_test_set(root_dir, preprocessed_data_dir)

    # Generate validation set from a subset of a random permutation of the training set
    train_path = os.path.join(preprocessed_data_dir, "test.txt")
    out_val_path = os.path.join(preprocessed_data_dir, "valid.txt")
    kgg.generate_val_set(train_path, out_val_path, val_ratio)
    