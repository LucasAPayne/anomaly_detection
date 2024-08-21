"""
A program that demonstrates templates knowledge graph generation from log files
as well as knowledge graph completion for anomaly detection
"""

import os
import yaml

from anomaly_detection.kg_generation import cyberml_dataset
from anomaly_detection.kg_completion.kg_completion import kg_completion

def main():
    """
    The demo code
    """
    labels = True
    cyberml_raw_data_dir = os.path.join(os.path.dirname(__file__), "data", "CyberML")
    cyberml_dataset.extract_dataset(cyberml_raw_data_dir, val_ratio=0.5, labels=labels)

    cfg_path = "config/cyberml.yaml"
    cfg: dict = {}
    with open(cfg_path, "r", encoding="utf-8") as cfg_file:
        cfg = yaml.safe_load(cfg_file)
    kg_completion(cfg)

if __name__ == "__main__":
    main()
