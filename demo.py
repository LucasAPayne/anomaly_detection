"""
A program that demonstrates templates knowledge graph generation from log files
as well as knowledge graph completion using both traditional and GNN models
"""

import os

import numpy as np

from anomaly_detection.kg_generation import ait_dataset
from anomaly_detection.kg_generation import cyberml_dataset
from anomaly_detection.kg_generation import kgg
from anomaly_detection.kg_completion import kgc

def main():
    """
    The demo code
    """
    np.random.seed(1234)
    # cyberml_root_dir = os.path.join(os.path.dirname(__file__), "data",
    #                                 "Industrial_Automation--Cybersecurity")
    # cyberml_dataset.extract_dataset(cyberml_root_dir, val_ratio=0.4)
    raw_data_dir = os.path.join(os.path.dirname(__file__), "data", "AIT-LDS-v1_1")
    labels = True
    exclude_errors = True
    ait_dataset.extract_dataset(raw_data_dir, exclude_errors)
    kgg.generate_kg(raw_data_dir, labels, gen_ids=True)

    # preprocessed_data_dir = os.path.join(raw_data_dir, "preprocessed")
    # train_path = os.path.join(preprocessed_data_dir, "train.txt")
    # test_path = os.path.join(preprocessed_data_dir, "test.txt")
    # val_path = os.path.join(preprocessed_data_dir, "valid.txt")
    # results_path = "results"
    # kgc.kgc(train_path, test_path, val_path, results_path, "AIT")

if __name__ == "__main__":
    main()
