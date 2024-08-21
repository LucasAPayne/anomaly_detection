import json
import os
import numpy as np

from pykeen.datasets import PathDataset
from pykeen.evaluation import LCWAEvaluationLoop, RankBasedEvaluator
from pykeen.models import ERModel, ComplEx, ConvE, DistMult
from pykeen.pipeline import pipeline
from pykeen.stoppers import EarlyStopper
from pykeen.losses import SoftplusLoss
from pykeen.sampling import BasicNegativeSampler

from torch.optim import Adam

# For reporting classification metrics
import sklearn

# TODO(lucas): Register datasets through PyKEEN
from .datasets.AIT import AIT
from .datasets.CyberML import CyberML
from .datasets.HDFS import HDFS

def report_classification_results(true_labels: list, pred_labels: list, file_path: str):
    label_names = ["normal", "suspicious"]
    accuracy = sklearn.metrics.accuracy_score(true_labels, pred_labels)
    precision, recall, f1_score, support = \
        sklearn.metrics.precision_recall_fscore_support(true_labels, pred_labels,
                                                        labels=label_names, pos_label="suspicious",
                                                        average="binary", zero_division=0)
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(true_labels, pred_labels).ravel()

    # Prevent divide by 0
    tpr = tp / (tp + fn) if tp + fn > 0 else 0.0
    tnr = tn / (tn + fp) if tn + fp > 0 else 0.0
    fpr = fp / (fp + tn) if fp + tn > 0 else 0.0
    fnr = fn / (fn + tp) if fn + tp > 0 else 0.0

    print(f"Accuracy: {accuracy}")
    print(f"F1-score: {f1_score}")
    print(f"precision: {precision}")
    print(f"recall: {recall}")
    print(f"support: {support}")
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"True Negatives: {tn}")
    print(f"False Negatives: {fn}")
    print(f"True Positive Rate: {tpr}")
    print(f"False Positive Rate: {fpr}")
    print(f"True Negative Rate: {tnr}")
    print(f"False Negative Rate: {fnr}")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"F1-score: {f1_score}\n")
        f.write(f"precision: {precision}\n")
        f.write(f"recall: {recall}\n")
        f.write(f"support: {support}\n")
        f.write(f"True Positives: {tp}\n")
        f.write(f"False Positives: {fp}\n")
        f.write(f"True Negatives: {tn}\n")
        f.write(f"False Negatives: {fn}\n")
        f.write(f"True Positive Rate: {tpr}\n")
        f.write(f"False Positive Rate: {fpr}\n")
        f.write(f"True Negative Rate: {tnr}\n")
        f.write(f"False Negative Rate: {fnr}\n")

def classify(model: ERModel, dataset: PathDataset, out_dir: str):
    log_ids: list[str] = dataset.metadata["test"]["log_ids"]
    labels: list[str] = dataset.metadata["test"]["labels"]
    has_log_ids = len(log_ids) > 0

    true_labels = []
    if has_log_ids:
        for log_id, label in zip(log_ids, labels):
            label_str = "suspicious" if label == "1" else "normal"
            true_labels.append((log_id, label_str))
    else:
        for label in labels:
            label_str = "suspicious" if label == "1" else "normal"
            true_labels.append((0, label_str))

    # TODO(lucas): Is there a better way to organize/extract labels and log IDs?
    # TODO(lucas): Make all these arrays np.array to begin with
    # NOTE(lucas): np.unique changes order, so the lists need to be sorted for comparison
    true_labels_np = np.array(true_labels)
    true_triple_labels: list = true_labels_np[:,1]

    pred_triple_labels = []

    confidence_cutoff = 0.5
    preds = model.predict_hrt(dataset.testing.mapped_triples)
    for pred, in preds:
        pred_label = "normal" if pred > confidence_cutoff else "suspicious"
        pred_triple_labels.append(pred_label)

    print("--------------------")
    print("Triple-Level Results")
    print("--------------------")
    out_file_triples = os.path.join(out_dir, "result_classification_triples.txt")
    report_classification_results(true_triple_labels, pred_triple_labels, out_file_triples)

    if has_log_ids:
        log_ids = true_labels_np[:,0]
        true_line_labels = np.copy(true_labels_np)
        true_line_labels[:,1] = "normal"
        true_line_labels_idx = np.unique(true_line_labels, return_index=True, axis=0)[1]
        true_line_labels = true_line_labels[true_line_labels_idx]

        # NOTE(lucas): For each true label, if the log ID has a suspicious label, find the log ID
        # in the unique array and change the label to suspicious
        for entry in true_labels:
            if entry[1] == "suspicious":
                idx = np.where(true_line_labels == entry[0])
                true_line_labels[idx,1] = "suspicious"

        # NOTE(lucas): Initialize predictions as all normal.
        # Then later, if any triple corresponding to a log ID is deemed suspicious,
        # change the prediction to suspicious
        pred_line_labels = np.copy(true_labels_np)
        pred_line_labels[:,1] = "normal"
        pred_line_labels_idx = np.unique(pred_line_labels, return_index=True, axis=0)[1]
        pred_line_labels = pred_line_labels[pred_line_labels_idx]
        pred_triple_labels = list(zip(log_ids, pred_triple_labels))

        pred_triple_labels_np = np.array(pred_triple_labels)
        for entry in pred_triple_labels_np:
            if entry[1] == "suspicious":
                idx = np.where(pred_line_labels == entry[0])
                pred_line_labels[idx,1] = "suspicious"

        print("--------------------")
        print("Line-Level Results")
        print("--------------------")
        out_file_lines = os.path.join(out_dir, "result_classification_lines.txt")
        report_classification_results(list(true_line_labels[:,1]), list(pred_line_labels[:,1]),
                                      out_file_lines)

def kg_completion(cfg: dict):
    dataset = None
    dataset_str = cfg["dataset"].lower()
    if dataset_str == "ait":
        dataset = AIT()
    elif dataset_str == "cyberml":
        dataset = CyberML()
    elif dataset_str == "hdfs":
        dataset = HDFS()

    training_triples_factory = dataset.training
    val_triples_factory = dataset.validation

    model = None
    kgc_model_str = cfg["model"].lower()
    if kgc_model_str == "complex":
        model = ComplEx(triples_factory=training_triples_factory,
                        embedding_dim=cfg["embedding_dim"],
                        random_seed=cfg["seed"]).cuda()
    elif kgc_model_str == "distmult":
        model = DistMult(triples_factory=training_triples_factory,
                        embedding_dim=cfg["embedding_dim"],
                        random_seed=cfg["seed"]).cuda()
    elif kgc_model_str == "conve":
        model = ConvE(triples_factory=training_triples_factory,
                      random_seed=cfg["seed"]).cuda()

    optimizer = Adam(params=model.get_grad_params(), lr=cfg["lr"],)
    negative_sampler = BasicNegativeSampler(mapped_triples=training_triples_factory.mapped_triples)
    loss = SoftplusLoss()
    evaluator = RankBasedEvaluator(batch_size=cfg["val_batch_size"], automatic_memory_optimization=False)

    # TODO(lucas): Use NopStopper if use_stopper is false?
    stopper = None
    if cfg["use_stopper"]:
        stopper = EarlyStopper(model, evaluator, training_triples_factory, val_triples_factory,
                               frequency=cfg["frequency"], patience=cfg["patience"],
                               metric=cfg["metric"])
        # NOTE(lucas): PyKEEN tries to override the evaluation batch size
        # on the first evaluation unless this value is set.
        stopper.evaluation_batch_size = evaluator.batch_size

    # TODO(lucas): Custom validation with classification
    # TODO(lucas): Save checkpoints
    # TODO(lucas): Replace pipeline with training/val loops?
    _ = pipeline(
        random_seed=cfg["seed"],
        dataset=dataset,
        model=model,
        loss=loss,
        negative_sampler=negative_sampler,
        negative_sampler_kwargs=dict(num_negs_per_pos=cfg["negative_samples"]),
        optimizer=optimizer,
        stopper=stopper,
        training_kwargs=dict(
            num_epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
        ),
    )

    test_loop = LCWAEvaluationLoop(model=model, triples_factory=dataset.testing,
                                   evaluator=evaluator)
    results = test_loop.evaluate(batch_size=cfg["val_batch_size"])
    # results = evaluator.evaluate(model=model, mapped_triples=dataset.testing.mapped_triples,
    #                              batch_size=cfg["val_batch_size"], automatic_memory_optimization=False)

    os.makedirs(cfg["out_dir"], exist_ok=True)
    kgc_result_path = os.path.join(cfg["out_dir"], "result_kgc.json")
    with open(kgc_result_path, "w", encoding="utf-8") as f:
        json.dump(results.to_dict(), f, indent=4)

    classify(model, dataset, cfg["out_dir"])
