import os
import yaml

from anomaly_detection.kg_generation import ait_dataset
from anomaly_detection.kg_generation.kg_generation import generate_kg
from anomaly_detection.kg_completion.kg_completion import kg_completion

def main():
    """
    The demo code
    """
    raw_data_dir = os.path.join(os.path.dirname(__file__), "data", "HDFS")
    exclude_errors = True
    ait_dataset.extract_dataset(raw_data_dir, exclude_errors)
    generate_kg(raw_data_dir, "HDFS")

    cfg_path = "config/hdfs.yaml"
    cfg: dict = {}
    with open(cfg_path, "r", encoding="utf-8") as cfg_file:
        cfg = yaml.safe_load(cfg_file)
    kg_completion(cfg)

if __name__ == "__main__":
    main()
