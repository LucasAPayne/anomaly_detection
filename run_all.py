"""
Run all combinations of models, datasets, and certain hyperparameters
"""
import ruamel.yaml

from anomaly_detection.kg_completion.kg_completion import kg_completion

def main():
    config_filename = "config/ait.yaml"
    dataset_dir = "data/AIT/preprocessed"
    # models = ["complex", "distmult", "ggnn_distmult", "gcn_distmult", "gcn_complex", "ggnn_complex"]
    # directions = ["undirected", "bi_sep", "bi_fuse"]
    models = ["complex", "gcn_complex", "ggnn_complex"]
    directions = ["undirected", "bi_sep", "bi_fuse"]

    yaml = ruamel.yaml.YAML()
    cfg = {}
    with open(config_filename, "r", encoding="utf-8") as config_file:
        cfg = yaml.load(config_file)

    for model in models:
        cfg["model"] = model
        for direction in directions:
            cfg["direction_option"] = direction
            with open(config_filename, "w", encoding="utf-8") as config_file:
                yaml.dump(cfg, config_file)
            kg_completion(config_filename, dataset_dir)

if __name__ == "__main__":
    main()
