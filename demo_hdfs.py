import cProfile
import io
import pstats
import os
import yaml

from anomaly_detection.kg_generation import hdfs_dataset
from anomaly_detection.kg_generation.kg_generation import generate_kg
from anomaly_detection.kg_completion.kg_completion import kg_completion

def main():
    """
    The demo code
    """
    raw_data_dir = os.path.join(os.path.dirname(__file__), "data", "HDFS")
    hdfs_dataset.extract_dataset(raw_data_dir)

    pr = cProfile.Profile()
    pr.enable()
    generate_kg(raw_data_dir, "HDFS")
    pr.disable()

    s = io.StringIO()
    sortby = pstats.SortKey.TIME
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()

    profile_path = os.path.join("results", "profile.txt")
    with open (profile_path, "w+", encoding="utf-8") as f:
        f.write(s.getvalue())

    cfg_path = "config/hdfs.yaml"
    cfg: dict = {}
    with open(cfg_path, "r", encoding="utf-8") as cfg_file:
        cfg = yaml.safe_load(cfg_file)
    kg_completion(cfg)

if __name__ == "__main__":
    main()
