"""
This module performs KGC on a dataset given train/test/validation files
and uses the PyKEEN module
"""
import json
import os

from pykeen.models import ConvE
from pykeen.training import SLCWATrainingLoop
from pykeen.triples import TriplesFactory
from pykeen.evaluation import LCWAEvaluationLoop
import pykeen.utils

import torch

def kgc(train_path: str, test_path: str, val_path: str=None,
        results_path: str=None, dataset_name: str=None, seed: int=12345) -> None:
    """
    Perform KGC on the dataset

    Parameters
    ----------
    - `train_path`: path to file containing training set
    - `test_path`: path to file containing testing set
    - `val_path`: path to file containing validation set
    - `results_path`: path to save model evaluation to
    - `dataset_name`: The name of the dataset, used for saving result file
    """
    pykeen.utils.set_random_seed(seed)
    training = TriplesFactory.from_path(train_path)
    testing = TriplesFactory.from_path(
            test_path,
            entity_to_id=training.entity_to_id,
            relation_to_id=training.relation_to_id)
    if val_path:
        validation = TriplesFactory.from_path(
                val_path,
                entity_to_id=training.entity_to_id,
                relation_to_id=training.relation_to_id)
    else:
        validation = None

    # Pick a model
    model = ConvE(triples_factory=training)

    # Pick an optimizer from Torch
    optimizer = torch.optim.Adam(params=model.get_grad_params())

    # Pick a training approach (sLCWA or LCWA)
    training_loop = SLCWATrainingLoop(
        model=model,
        triples_factory=training,
        optimizer=optimizer,
    )

    _ = training_loop.train(
        triples_factory=training,
        num_epochs=5,
        batch_size=256,
        callbacks="evaluation-loop",
        callback_kwargs=dict(
            prefix="validation",
            factory=validation,
        ),
    )

    # Pick an evaluation loop
    evaluation_loop = LCWAEvaluationLoop(
        model=model,
        triples_factory=testing,
    )

    # Evaluate
    results = evaluation_loop.evaluate()
    with open(os.path.join(results_path, f"{dataset_name}_results.json"), "w", encoding="utf-8")\
        as outfile:
        json.dump(results.to_dict(), outfile, indent=4, separators=(",", ": "))
