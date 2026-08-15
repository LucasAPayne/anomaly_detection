"""
Check the performance of the named entity recognition and triple generation tasks of the
LLM-assisted knowledge graph generation process.
Results are reported with precision, recall, and F1-score.
"""

import argparse
import logging
import os
import subprocess
import sys
import time

from dataclasses import dataclass

import orjson

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from anomaly_detection.kg_generation import ait_dataset
from anomaly_detection.kg_generation.llm import find_unique_log_formats, format_seconds

@dataclass(frozen=True)
class FrozenEntity:
    """The same as `ExtractedEntity`, but frozen to be hashable"""
    text: str
    type: str
    start: int
    end: int

@dataclass
class ConfusionMatrix:
    """True/false positive/negative"""
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0

    def __add__(self, other):
        return ConfusionMatrix(
            self.tp + other.tp, self.tn + other.tn, self.fp + other.fp, self.fn + other.fn
        )

@dataclass
class AccuracyMetrics:
    """Class for storing accuracy and F1-score metrics"""
    precision: float = 0.0
    recall:    float = 0.0
    f1:        float = 0.0
    accuracy:  float = 0.0

def print_metrics(mat: ConfusionMatrix, metrics: AccuracyMetrics) -> None:
    print("TP:", mat.tp, "FP:", mat.fp, "FN:", mat.fn)
    print("Precision:", metrics.precision)
    print("Recall:", metrics.recall)
    print("F1:", metrics.f1)
    print("Accuracy:", metrics.accuracy)

def confusion_matrix_ner(
        gold: list[FrozenEntity],
        pred: list[FrozenEntity]
) -> ConfusionMatrix:
    """
    Compute the confusion matrix for a list of predicted and gold entities.
    True negatives are not considered because that space is too large.
    (Almost anything could be a true negative.)
    """
    gold_set = set(gold)
    pred_set = set(pred)

    result = ConfusionMatrix()
    result.tp = len(gold_set & pred_set)
    result.fp = len(pred_set - gold_set)
    result.fn = len(gold_set - pred_set)

    return result

def confusion_matrix_triples(
    gold: set[tuple[str, str, str]],
    pred: set[tuple[str, str, str]]
) -> ConfusionMatrix:
    """
    Compute the confusion matrix for a list of predicted and gold triples.
    True negatives are not considered because that space is too large.
    (Almost anything could be a true negative.)
    """
    result = ConfusionMatrix()
    result.tp = len(gold & pred)
    result.fp = len(pred - gold)
    result.fn = len(gold - pred)

    return result

def compute_metrics(tp, fn, fp) -> AccuracyMetrics:
    """ Compute precision, recall, F1, and accuracy metrics. """
    # If any metric calculation would divide by 0, that metric becomes 0.
    precision = tp / (tp + fp) if tp + fp else 0
    recall    = tp / (tp + fn) if tp + fn else 0
    f1        = 2 * precision * recall / (precision + recall) if precision + recall else 0
    accuracy = tp / (tp + fn + fp) if tp + fn + fp else 0

    result = AccuracyMetrics(precision, recall, f1, accuracy)
    return result

def run_ait_test(root_dir: str, current_dir: str, gen_templates_path: str) -> None:
    """ Process logs from the AIT dataset. """
    start = time.time()

    raw_data_dir = os.path.join(root_dir, "data", "AIT")
    preprocessed_data_dir = os.path.join(raw_data_dir, "preprocessed")
    unique_logs_path = os.path.join(preprocessed_data_dir, "unique_logs.txt")
    unique_templates_path = os.path.join(preprocessed_data_dir, "unique_templates.jsonl")

    if not os.path.exists(unique_logs_path) or not os.path.exists(unique_templates_path):
        log_dir = os.path.join(raw_data_dir, "data")
        log_list = ait_dataset.gather_files(log_dir)
        for i, log in enumerate(log_list):
            log_list[i] = os.path.join(log_dir, log)
        find_unique_log_formats(log_list, preprocessed_data_dir)

    llm_config_path = os.path.join(root_dir, "config", "llm_config.yaml")
    valid_types_path = os.path.join(current_dir, "valid_types.json")
    valid_rels_path = os.path.join(current_dir, "valid_rels.json")

    subprocess.run([
        "mpirun",
        sys.executable,
        "-m",
        "anomaly_detection.kg_generation.llm",
        "--log-path", unique_logs_path,
        "--template-path", unique_templates_path,
        "--config-path", llm_config_path,
        "--valid-types-path", valid_types_path,
        "--valid-rels-path", valid_rels_path,
        "--out-path", gen_templates_path
    ],
    check=True,
    text=True)

    logging.info(f"AIT run completed in {format_seconds(time.time() - start)}")

def resolve_triples(template: dict) -> set[tuple[str, str, str]]:
    entities = template["entities"]

    return {
        (
            entities[triple["subject"]]["text"],
            triple["relation"],
            entities[triple["object"]]["text"],
        )
        for triple in template["triples"]
    }

def main(iteration: int):
    # Generate templates from the AIT dataset
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(current_dir, "..")
    ait_gen_templates_path = os.path.join(current_dir, "results", "AIT", f"ait_accuracy_test_templates_{iteration}.json")
    ait_answer_path = os.path.join(current_dir, "ait_template_labels.json")

    logging.info("### AIT ###")
    run_ait_test(root_dir, current_dir, ait_gen_templates_path)

    # TODO(lucas): A hack for now since llm.py does not allow an easy way to rename run_log
    run_log_path_old = os.path.join(current_dir, "results", "run_log.json")
    run_log_path_new = os.path.join(current_dir, "results", f"run_log_{iteration}.json")
    os.rename(run_log_path_old, run_log_path_new)

    # Get the templates predicted from the AIT test and the gold templates from the answer file
    with open(ait_gen_templates_path, "rb") as templates_file:
        file_contents = templates_file.read()
        pred_templates = orjson.loads(file_contents)["templates"]

    with open(ait_answer_path, "rb") as answer_file:
        file_contents = answer_file.read()
        gold_templates = orjson.loads(file_contents)["templates"]

    if len(pred_templates) != len(gold_templates):
        logging.error(f"{len(pred_templates)} generated, but answer file has {len(gold_templates)} answers")
        return

    ner_mat = ConfusionMatrix()
    triple_mat = ConfusionMatrix()
    for gold_template, pred_template in zip(gold_templates, pred_templates):
        pred_entities = []
        for e in pred_template["entities"]:
            pred_entities.append(FrozenEntity(e["text"], e["type"], e["start"], e["end"]))

        gold_entities = []
        for e in gold_template["entities"]:
            gold_entities.append(FrozenEntity(e["text"], e["type"], e["start"], e["end"]))

        ner_mat += confusion_matrix_ner(gold_entities, pred_entities)

        # Compute metrics for triple generation task
        gold_triples = resolve_triples(gold_template)
        pred_triples = resolve_triples(pred_template)
        triple_mat += confusion_matrix_triples(gold_triples, pred_triples)

    ner_metrics = compute_metrics(ner_mat.tp, ner_mat.fp, ner_mat.fn)
    triple_metrics = compute_metrics(triple_mat.tp, triple_mat.fp, triple_mat.fn)

    print("### NER Metrics ###")
    print_metrics(ner_mat, ner_metrics)
    print("\n")
    print("### Triple Metrics ###")
    print_metrics(triple_mat, triple_metrics)
    print("\n\n")

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    log_format = "[%(asctime)s]: %(name)s: %(levelname)s: %(message)s"
    logging.basicConfig(
        level=logging.DEBUG,
        format=log_format
    )
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)

    parser = argparse.ArgumentParser(
        description="Run an accuracy test for LLM template generation for KG generation"
    )
    parser.add_argument("-i", "--iterations", type=int, required=True,
                        help="The number of times to run the test")
    args = parser.parse_args()

    for i in range(args.iterations):
        main(i)
