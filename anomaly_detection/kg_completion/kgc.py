import json

from pykeen.models import ConvE
from pykeen.training import SLCWATrainingLoop
from pykeen.triples import TriplesFactory
from pykeen.evaluation import LCWAEvaluationLoop

import torch

training = TriplesFactory.from_path("data/AIT/train.ttl")
testing = TriplesFactory.from_path(
        "data/AIT/test.ttl",
        entity_to_id=training.entity_to_id,
        relation_to_id=training.relation_to_id)
validation = TriplesFactory.from_path(
        "data/AIT/val.ttl",
        entity_to_id=training.entity_to_id,
        relation_to_id=training.relation_to_id)

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
with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results.to_dict(), f, indent=4, separators=(",", ": "))
