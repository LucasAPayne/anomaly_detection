"""
Run all combinations of models, datasets, and certain hyperparameters
"""
import numpy as np
import ruamel.yaml
import torch

from anomaly_detection.kg_completion.kg_completion import kg_completion

def main():
    config_filename = "config/ait.yaml"
    dataset_dir = "data/AIT/preprocessed"
    models = ["complex", "distmult", "ggnn_distmult", "gcn_distmult", "gcn_complex", "ggnn_complex"]
    directions =  ["undirected", "bi_sep", "bi_fuse"]
    n_iter = 10
    rng = np.random.default_rng()

    param_space = {
        "l2": rng.uniform(0.0, 0.1, n_iter).tolist(),
        "label_smoothing": rng.uniform(0.0, 0.1, n_iter).tolist(),
        "input_drop": rng.uniform(0.0, 0.5, n_iter).tolist(),
        "feat_drop": rng.uniform(0.0, 0.5, n_iter).tolist()
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
