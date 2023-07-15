"""
Run all combinations of models, datasets, and certain hyperparameters
"""
import numpy as np
import ruamel.yaml
import torch

from anomaly_detection.kg_completion.kg_completion import kg_completion

def main():
    np.random.seed(1234)

    config_filename = "config/cyberml.yaml"
    dataset_dir = "data/CyberML/preprocessed"
    models = ["gcn_complex", "ggnn_complex", "complex", "gcn_distmult", "ggnn_distmult", "distmult"]

    yaml = ruamel.yaml.YAML()
    cfg = {}
    with open(config_filename, "r", encoding="utf-8") as config_file:
        cfg = yaml.load(config_file)

    for model in models:
        cfg["model"] = model
        if model == "gcn_distmult" or model == "gnn_distmult":
            cfg["direction_option"] = "bi_sep"
        elif model == "gcn_complex":
            cfg["direction_option"] = "undirected"
        elif model == "ggnn_complex":
            cfg["direction_option"] = "bi_fuse"

        with open(config_filename, "w", encoding="utf-8") as config_file:
            yaml.dump(cfg, config_file)

        kg_completion(config_filename, dataset_dir)

    config_filename = "config/ait.yaml"
    dataset_dir = "data/AIT/preprocessed"
    models = ["complex", "distmult", "ggnn_distmult", "gcn_distmult", "gcn_complex", "ggnn_complex"]
    directions =  ["undirected", "bi_sep", "bi_fuse"]
    n_iter = 10
    rng = np.random.default_rng()

    param_space = {
        "l2": rng.uniform(0.0, 0.1, n_iter).round(4).tolist(),
        "label_smoothing": rng.uniform(0.0, 0.1, n_iter).round(4).tolist(),
        "input_drop": rng.uniform(0.0, 0.5, n_iter).round(4).tolist(),
        "feat_drop": rng.uniform(0.0, 0.5, n_iter).round(4).tolist()
    }

    yaml = ruamel.yaml.YAML()
    cfg = {}
    with open(config_filename, "r", encoding="utf-8") as config_file:
        cfg = yaml.load(config_file)

    # Detect whether GPU is available and use if it is
    if torch.cuda.is_available():
        cfg["cuda"] = "true"
        cfg["gpu"] = torch.cuda.current_device()

    for model in models:
        cfg["model"] = model
        for direction in directions:
            cfg["direction_option"] = direction

            for i in range(n_iter):
                for key, value in param_space.items():
                    cfg[key] = value[i]

                with open(config_filename, "w", encoding="utf-8") as config_file:
                    yaml.dump(cfg, config_file)
                kg_completion(config_filename, dataset_dir)

if __name__ == "__main__":
    main()
