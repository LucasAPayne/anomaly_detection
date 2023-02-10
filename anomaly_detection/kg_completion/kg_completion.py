import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader

from graph4nlp.pytorch.modules.utils.config_utils import get_yaml_config
from graph4nlp.pytorch.datasets.kinship import KinshipDataset
from graph4nlp.pytorch.modules.utils.logger import Logger

from .model import Complex, ConvE, Distmult, GCNComplex, GCNDistMult,GGNNComplex, GGNNDistMult

def ranking_and_hits_this(cfg, model, dev_rank_batcher, vocab, name, kg_graph=None, logger=None):
    print("")
    print("-" * 50)
    print(name)
    print("-" * 50)
    print("")
    if logger is not None:
        logger.write("")
        logger.write("-" * 50)
        logger.write(name)
        logger.write("-" * 50)
        logger.write("")
    hits_left = []
    hits_right = []
    hits = []
    ranks = []
    ranks_left = []
    ranks_right = []
    for _ in range(10):
        hits_left.append([])
        hits_right.append([])
        hits.append([])

    for i, str2var in enumerate(dev_rank_batcher):
        e1 = str2var["e1_tensor"]
        e2 = str2var["e2_tensor"]
        rel = str2var["rel_tensor"]
        rel_reverse = str2var["rel_eval_tensor"]
        e2_multi1 = str2var["e2_multi1"].float()
        e2_multi2 = str2var["e2_multi2"].float()
        if cfg["cuda"]:
            e1 = e1.to("cuda")
            e2 = e2.to("cuda")
            rel = rel.to("cuda")
            rel_reverse = rel_reverse.to("cuda")
            e2_multi1 = e2_multi1.to("cuda")
            e2_multi2 = e2_multi2.to("cuda")

        pred1 = model(e1, rel, kg_graph)
        pred2 = model(e2, rel_reverse, kg_graph)
        pred1, pred2 = pred1.data, pred2.data
        e1, e2 = e1.data, e2.data
        e2_multi1, e2_multi2 = e2_multi1.data, e2_multi2.data
        for i in range(e1.shape[0]):
            # these filters contain ALL labels
            filter1 = e2_multi1[i].long()
            filter2 = e2_multi2[i].long()

            # save the prediction that is relevant
            target_value1 = pred1[i, e2[i, 0].item()].item()
            target_value2 = pred2[i, e1[i, 0].item()].item()
            # zero all known cases (this are not interesting)
            # this corresponds to the filtered setting
            pred1[i][filter1] = 0.0
            pred2[i][filter2] = 0.0
            # write base the saved values
            pred1[i][e2[i]] = target_value1
            pred2[i][e1[i]] = target_value2

        # sort and rank
        max_values, argsort1 = torch.sort(pred1, 1, descending=True)
        max_values, argsort2 = torch.sort(pred2, 1, descending=True)

        argsort1 = argsort1.cpu().numpy()
        argsort2 = argsort2.cpu().numpy()
        for i in range(e1.shape[0]):
            # find the rank of the target entities
            rank1 = np.where(argsort1[i] == e2[i, 0].item())[0][0]
            rank2 = np.where(argsort2[i] == e1[i, 0].item())[0][0]
            # rank+1, since the lowest rank is rank 1 not rank 0
            ranks.append(rank1 + 1)
            ranks_left.append(rank1 + 1)
            ranks.append(rank2 + 1)
            ranks_right.append(rank2 + 1)

            # this could be done more elegantly, but here you go
            for hits_level in range(10):
                if rank1 <= hits_level:
                    hits[hits_level].append(1.0)
                    hits_left[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)
                    hits_left[hits_level].append(0.0)

                if rank2 <= hits_level:
                    hits[hits_level].append(1.0)
                    hits_right[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)
                    hits_right[hits_level].append(0.0)

        # dev_rank_batcher.state.loss = [0]

    for i in range(10):
        print("Hits left @{0}: {1}".format(i + 1, np.mean(hits_left[i])))
        print("Hits right @{0}: {1}".format(i + 1, np.mean(hits_right[i])))
        print("Hits @{0}: {1}".format(i + 1, np.mean(hits[i])))
    print("Mean rank left: {0}".format(np.mean(ranks_left)))
    print("Mean rank right: {0}".format(np.mean(ranks_right)))
    print("Mean rank: {0}".format(np.mean(ranks)))
    print("Mean reciprocal rank left: {0}".format(np.mean(1.0 / np.array(ranks_left))))
    print("Mean reciprocal rank right: {0}".format(np.mean(1.0 / np.array(ranks_right))))
    print("Mean reciprocal rank: {0}".format(np.mean(1.0 / np.array(ranks))))

    if logger is not None:
        for i in [0, 9]:
            logger.write("Hits left @{0}: {1}".format(i + 1, np.mean(hits_left[i])))
            logger.write("Hits right @{0}: {1}".format(i + 1, np.mean(hits_right[i])))
            logger.write("Hits @{0}: {1}".format(i + 1, np.mean(hits[i])))
        logger.write("Mean rank left: {0}".format(np.mean(ranks_left)))
        logger.write("Mean rank right: {0}".format(np.mean(ranks_right)))
        logger.write("Mean rank: {0}".format(np.mean(ranks)))
        logger.write("Mean reciprocal rank left: {0}".format(np.mean(1.0 / np.array(ranks_left))))
        logger.write("Mean reciprocal rank right: {0}".format(np.mean(1.0 / np.array(ranks_right))))
        logger.write("Mean reciprocal rank: {0}".format(np.mean(1.0 / np.array(ranks))))

    return np.mean(1.0 / np.array(ranks))

class KGC(nn.Module):
    def __init__(self, cfg, num_entities, num_relations):
        super(KGC, self).__init__()
        self.cfg = cfg
        self.num_entities = num_entities
        self.num_relations = num_relations
        if cfg["model"] is None:
            model = ConvE(cfg, num_entities, num_relations)
        elif cfg["model"] == "conve":
            model = ConvE(cfg, num_entities, num_relations)
        # elif cfg["model"] == "ggnn_conve":
        #     model = GGNNConvE(cfg, num_entities, num_relations)
        # elif cfg["model"] == "gcn_conve":
        #     model = GCNConvE(cfg, num_entities, num_relations)
        elif cfg["model"] == "distmult":
            model = Distmult(cfg, num_entities, num_relations)
        elif cfg["model"] == "complex":
            model = Complex(cfg, num_entities, num_relations)
        elif cfg["model"] == "ggnn_distmult":
            model = GGNNDistMult(cfg, num_entities, num_relations)
        elif cfg["model"] == "gcn_distmult":
            model = GCNDistMult(cfg, num_entities, num_relations)
        elif cfg["model"] == "ggnn_complex":
            model = GGNNComplex(cfg, num_entities, num_relations)
        elif cfg["model"] == "gcn_complex":
            model = GCNComplex(cfg, num_entities, num_relations)
        else:
            raise Exception("Unknown model type!")

        self.model = model

    def init(self):
        return self.model.init()

    def forward(self, e1_tensor, rel_tensor, KG_graph):
        return self.model(e1_tensor, rel_tensor, KG_graph)

    def loss(self, pred, e2_multi):
        return self.model.loss(pred, e2_multi)

    def inference_forward(self, collate_data, KG_graph):
        e1_tensor = collate_data["e1_tensor"]
        rel_tensor = collate_data["rel_tensor"]
        if self.cfg["cuda"]:
            e1_tensor = e1_tensor.to("cuda")
            rel_tensor = rel_tensor.to("cuda")
        return self.model(e1_tensor, rel_tensor, KG_graph)

    def post_process(self, logits, e2=None):
        max_values, argsort1 = torch.sort(logits, 1, descending=True)
        rank1 = np.where(argsort1.cpu().numpy()[0] == e2[0, 0].item())[0][0]

        print("ground truth e2 rank = {}".format(rank1 + 1))

        # argsort1 = argsort1.cpu().numpy()
        return argsort1[:, 0].item()

def kg_completion(config_path: str, dataset_dir: str) -> None:
    if not os.path.exists("saved_models"):
        os.mkdir("saved_models")
    cfg = get_yaml_config(config_path)

    model_name = "{0}_{1}_{2}_{3}_{4}_{5}_{6}".format(
        cfg["model"],
        cfg["direction_option"],
        cfg["l2"],
        cfg["label_smoothing"],
        cfg["input_drop"],
        cfg["hidden_drop"],
        cfg["feat_drop"]
    )
    model_path = "saved_models/{0}_{1}.model".format(
        cfg["dataset"], model_name
    )

    torch.manual_seed(cfg["seed"])

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    np.set_printoptions(precision=3)
    cudnn.benchmark = True

    dataset = KinshipDataset(
        root_dir=dataset_dir,
        topology_subdir="kgc",
    )

    cfg["out_dir"] = os.path.join(cfg["out_dir"], "{0}_{1}".format(cfg["dataset"], model_name))

    logger = Logger(
        cfg["out_dir"],
        config={k: v for k, v in cfg.items() if k != "device"},
        overwrite=True,
    )
    logger.write(cfg["out_dir"])

    train_dataloader = DataLoader(
        dataset.train,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["loader_threads"],
        collate_fn=dataset.collate_fn,
    )
    val_dataloader = DataLoader(
        dataset.val,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["loader_threads"],
        collate_fn=dataset.collate_fn,
    )
    test_dataloader = DataLoader(
        dataset.test,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["loader_threads"],
        collate_fn=dataset.collate_fn,
    )

    data = []
    rows = []
    columns = []
    num_entities = len(dataset.vocab_model.in_word_vocab)
    num_relations = len(dataset.vocab_model.out_word_vocab)

    if cfg["preprocess"]:
        for i, str2var in enumerate(train_dataloader):
            print("batch number:", i)
            for j in range(str2var["e1"].shape[0]):
                for k in range(str2var["e2_multi1"][j].shape[0]):
                    if str2var["e2_multi1"][j][k] != 0:
                        data.append(str2var["rel"][j].tolist()[0])
                        rows.append(str2var["e1"][j].tolist()[0])
                        columns.append(str2var["e2_multi1"][j][k].tolist())
                    else:
                        break

        from graph4nlp.pytorch.data.data import GraphData

        KG_graph = GraphData()
        KG_graph.add_nodes(num_entities)
        for e1, rel, e2 in zip(rows, data, columns):
            KG_graph.add_edge(e1, e2)
            eid = KG_graph.edge_ids(e1, e2)[0]
            KG_graph.edge_attributes[eid]["token"] = rel

        torch.save(KG_graph, os.path.join(dataset_dir, "processed", "kgc", "KG_graph.pt"))
    else:
        graph_path = os.path.join(dataset_dir, "processed", "kgc", "KG_graph.pt")
        KG_graph = torch.load(graph_path)

    if cfg["cuda"] is True:
        KG_graph = KG_graph.to("cuda")
    else:
        KG_graph = KG_graph.to("cpu")

    model = KGC(cfg, num_entities, num_relations)

    if cfg["cuda"] is True:
        model.to("cuda")

    if cfg["resume"]:
        model_params = torch.load(model_path)
        print(model)
        total_param_size = []
        params = [(key, value.size(), value.numel()) for key, value in model_params.items()]
        for key, size, count in params:
            total_param_size.append(count)
            print(key, size, count)
        print(np.array(total_param_size).sum())
        model.load_state_dict(model_params)
        model.eval()
        ranking_and_hits_this(
            cfg,
            model,
            test_dataloader,
            dataset.vocab_model,
            "test_evaluation",
            kg_graph=KG_graph,
            logger=logger,
        )
        ranking_and_hits_this(
            cfg,
            model,
            val_dataloader,
            dataset.vocab_model,
            "dev_evaluation",
            kg_graph=KG_graph,
            logger=logger,
        )
    else:
        model.init()

    # total_param_size = []
    # params = [value.numel() for value in model.parameters()]
    # print(params)
    # print(np.sum(params))

    best_mrr = 0

    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["l2"])
    for epoch in range(cfg["epochs"]):
        model.train()
        for str2var in train_dataloader:
            opt.zero_grad()
            e1_tensor = str2var["e1_tensor"]
            rel_tensor = str2var["rel_tensor"]
            e2_multi = str2var["e2_multi1_binary"].float()
            if cfg["cuda"]:
                e1_tensor = e1_tensor.to("cuda")
                rel_tensor = rel_tensor.to("cuda")
                e2_multi = e2_multi.to("cuda")
            # label smoothing
            e2_multi = ((1.0 - cfg["label_smoothing"]) * e2_multi) + (1.0 / e2_multi.size(1))

            pred = model(e1_tensor, rel_tensor, KG_graph)
            loss = model.loss(pred, e2_multi)
            loss.backward()
            opt.step()

            # train_batcher.state.loss = loss.cpu()

        model.eval()
        with torch.no_grad():
            if epoch % 2 == 0 and epoch > 0:
                dev_mrr = ranking_and_hits_this(
                    cfg,
                    model,
                    val_dataloader,
                    dataset.vocab_model,
                    "dev_evaluation",
                    kg_graph=KG_graph,
                    logger=logger,
                )
                if dev_mrr > best_mrr:
                    best_mrr = dev_mrr
                    logger.write("Best MRR")
                    print("saving best model to {0}".format(model_path))
                    torch.save(model.state_dict(), model_path)
            if epoch % 2 == 0:
                if epoch > 0:
                    ranking_and_hits_this(
                        cfg,
                        model,
                        test_dataloader,
                        dataset.vocab_model,
                        "test_evaluation",
                        kg_graph=KG_graph,
                        logger=logger,
                    )
