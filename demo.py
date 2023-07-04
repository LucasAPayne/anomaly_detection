"""
A program that demonstrates templates knowledge graph generation from log files
as well as knowledge graph completion using both traditional and GNN models
"""

import os
import numpy as np

from anomaly_detection.kg_generation import cyberml_dataset
from anomaly_detection.kg_generation.kg_generation import generate_kg
from anomaly_detection.kg_completion.wrangle_KG import wrangle_kg
from anomaly_detection.kg_completion.kg_completion import kg_completion
from collect_results import collect_results, find_best_model

def main():
    """
    The demo code
    """
    np.random.seed(1234)
    # ait_raw_data_dir = os.path.join(os.path.dirname(__file__), "data", "AIT")
    # ait_preprocessed_data_dir = os.path.join(ait_raw_data_dir, "preprocessed")
    # labels = True
    # exclude_errors = True
    # ait_dataset.extract_dataset(ait_raw_data_dir, exclude_errors)
    # generate_kg(ait_raw_data_dir, labels, gen_ids=True)

    # wrangle_kg(ait_preprocessed_data_dir)
    # kg_completion("config/ait.yaml", "data/AIT/preprocessed")
    # collect_results("results/kgc")
    # find_best_model("results/kgc/AIT")

    labels = True
    cyberml_raw_data_dir = os.path.join(os.path.dirname(__file__), "data", "CyberML")
    cyberml_preprocessed_data_dir = os.path.join(cyberml_raw_data_dir, "preprocessed")
    cyberml_dataset.extract_dataset(cyberml_raw_data_dir, val_ratio=0.5, labels=labels)
    wrangle_kg(cyberml_preprocessed_data_dir, labels=labels)
    kg_completion("config/cyberml.yaml", "data/CyberML/preprocessed", labels=labels)

if __name__ == "__main__":
    main()
