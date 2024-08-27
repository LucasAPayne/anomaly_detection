"""
A program that demonstrates templates knowledge graph generation from log files
as well as knowledge graph completion for anomaly detection
"""

import cProfile
import io
import pstats
import os
import yaml

from anomaly_detection.kg_generation import ait_dataset
from anomaly_detection.kg_generation.kg_generation import generate_kg
from anomaly_detection.kg_completion.kg_completion import kg_completion

def main():
    """
    The demo code
    """
    ait_raw_data_dir = os.path.join(os.path.dirname(__file__), "data", "AIT")
    exclude_errors = True
    ait_dataset.extract_dataset(ait_raw_data_dir, exclude_errors)

    pr = cProfile.Profile()
    pr.enable()
    generate_kg(ait_raw_data_dir, "AIT")
    pr.disable()

    s = io.StringIO()
    sortby = pstats.SortKey.TIME
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()

    with open ("profile.txt", "w+", encoding="utf-8") as f:
        f.write(s.getvalue())

    # cfg_path = "config/ait.yaml"
    # cfg: dict = {}
    # with open(cfg_path, "r", encoding="utf-8") as cfg_file:
    #     cfg = yaml.safe_load(cfg_file)
    # kg_completion(cfg)

if __name__ == "__main__":
    main()
