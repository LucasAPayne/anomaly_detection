import os

from pykeen.hpo import hpo_pipeline_from_config

from .datasets.AIT import AIT_TRAIN_PATH, AIT_TEST_PATH, AIT_VALIDATE_PATH
from .datasets.CyberML import CYBERML_TRAIN_PATH, CYBERML_TEST_PATH, CYBERML_VALIDATE_PATH
from .datasets.HDFS import HDFS_TRAIN_PATH, HDFS_TEST_PATH, HDFS_VALIDATE_PATH

def main():
    dataset = "cyberml"
    model = "complex"

    train_path = None
    test_path = None
    val_path = None

    dataset = dataset.lower()
    if dataset == "ait":
        train_path = AIT_TRAIN_PATH
        test_path = AIT_TEST_PATH
        val_path = AIT_VALIDATE_PATH
    elif dataset == "cyberml":
        train_path = CYBERML_TRAIN_PATH
        test_path = CYBERML_TEST_PATH
        val_path = CYBERML_VALIDATE_PATH
    elif dataset == "hdfs":
        train_path = HDFS_TRAIN_PATH
        test_path = HDFS_TEST_PATH
        val_path = HDFS_VALIDATE_PATH

    config = {
        "optuna": dict(n_trials=50),
        "pipeline": dict(
            training=train_path,
            testing=test_path,
            validation=val_path,
            model=model
        )
    }

    result = hpo_pipeline_from_config(config)

    out_dir = os.path.join("results", dataset, model)
    os.makedirs(out_dir, exist_ok=True)
    result.save_to_directory(out_dir)

if __name__ == "__main__":
    main()
