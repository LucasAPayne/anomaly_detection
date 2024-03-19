import os

from anomaly_detection.kg_generation import hdfs_dataset
from anomaly_detection.kg_generation.kg_generation import generate_kg
# from anomaly_detection.kg_completion.wrangle_KG import wrangle_kg
# from anomaly_detection.kg_completion.kg_completion import kg_completion
# from collect_results import collect_results, find_best_model

def main():
    raw_data_dir = os.path.join(os.path.dirname(__file__), "data", "HDFS")
    preprocessed_data_dir = os.path.join(raw_data_dir, "preprocessed")
    hdfs_dataset.extract_dataset(raw_data_dir, chunk_size=10000)
    generate_kg(raw_data_dir, "HDFS", gen_ids=False)

    # wrangle_kg(preprocessed_data_dir, labels=True)
    # kg_completion("config/hdfs.yaml", "data/HDFS/preprocessed", labels=True)
    # collect_results("results/kgc/HDFS")
    # find_best_model("results/kgc/HDFS")

if __name__ == "__main__":
    main()
