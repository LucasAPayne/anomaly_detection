from __future__ import print_function
import json
import os
import numpy as np

def write_training_graph(cases, graph, path):
    with open(path, "w") as f:
        for key in graph:
            e1, rel = key
            # (Mike, fatherOf, John)
            # (John, fatherOf, Tom)
            # (John, fatherOf_reverse, Mike)
            # (Tom, fatherOf_reverse, John)

            # (John, fatherOf) -> Tom
            # (John, fatherOf_reverse, Mike)
            entities1 = " ".join(list(graph[key]))

            data_point = {}
            data_point["e1"] = e1
            data_point["e2"] = "None"
            data_point["rel"] = rel
            data_point["rel_eval"] = "None"
            data_point["e2_multi1"] = entities1
            data_point["e2_multi2"] = "None"
            data_point["label"] = 1

            f.write(json.dumps(data_point) + "\n")


def write_evaluation_graph(cases, graph, path):
    with open(path, "w") as f:
        n1 = 0
        n2 = 0
        for e1, rel, e2, label in cases:
            # (Mike, fatherOf) -> John
            # (John, fatherOf, Tom)
            rel_reverse = rel + "_reverse"
            entities1 = " ".join(list(graph[(e1, rel)]))
            entities2 = " ".join(list(graph[(e2, rel_reverse)]))

            n1 += len(entities1.split(" "))
            n2 += len(entities2.split(" "))

            data_point = {}
            data_point["e1"] = e1
            data_point["e2"] = e2
            data_point["rel"] = rel
            data_point["rel_eval"] = rel_reverse
            data_point["e2_multi1"] = entities1
            data_point["e2_multi2"] = entities2
            data_point["label"] = label

            f.write(json.dumps(data_point) + "\n")


def wrangle_kg(data_dir, labels=False):
    np.random.RandomState(234234)

    files = ["train.txt", "valid.txt", "test.txt"]
    data = []
    label_graph = {}
    train_graph = {}
    test_cases = {}
    for file in files:
        test_cases[file] = []
        train_graph[file] = {}
        with open(os.path.join(data_dir, file), "r", encoding="utf-8") as infile:
            data = infile.readlines() + data

        with open(os.path.join(data_dir, file), "r", encoding="utf-8") as infile:
            for line in infile:
                if labels:
                    e1, rel, e2, label = line.rstrip().split('\t')
                else:
                    e1, rel, e2 = line.split('\t')

                e1 = e1.strip()
                e2 = e2.strip()
                rel = rel.strip()
                rel_reverse = rel + "_reverse"

                # data
                # (Mike, fatherOf, John)
                # (John, fatherOf, Tom)

                if (e1, rel) not in label_graph:
                    label_graph[(e1, rel)] = set()

                if (e2, rel_reverse) not in label_graph:
                    label_graph[(e2, rel_reverse)] = set()

                if (e1, rel) not in train_graph[file]:
                    train_graph[file][(e1, rel)] = set()
                if (e2, rel_reverse) not in train_graph[file]:
                    train_graph[file][(e2, rel_reverse)] = set()

                # labels
                # (Mike, fatherOf, John)
                # (John, fatherOf, Tom)
                # (John, fatherOf_reverse, Mike)
                # (Tom, fatherOf_reverse, Mike)
                label_graph[(e1, rel)].add(e2)

                label_graph[(e2, rel_reverse)].add(e1)

                # test cases
                # (Mike, fatherOf, John)
                # (John, fatherOf, Tom)
                test_cases[file].append([e1, rel, e2, label])

                # data
                # (Mike, fatherOf, John)
                # (John, fatherOf, Tom)
                # (John, fatherOf_reverse, Mike)
                # (Tom, fatherOf_reverse, John)
                train_graph[file][(e1, rel)].add(e2)
                train_graph[file][(e2, rel_reverse)].add(e1)

    all_cases = test_cases["train.txt"] + test_cases["valid.txt"] + test_cases["test.txt"]
    write_training_graph(
        test_cases["train.txt"],
        train_graph["train.txt"],
        os.path.join(data_dir, "e1rel_to_e2_train.json")
    )
    write_evaluation_graph(
        test_cases["valid.txt"],
        label_graph,
        os.path.join(data_dir, "e1rel_to_e2_ranking_dev.json")
    )
    write_evaluation_graph(
        test_cases["test.txt"],
        label_graph,
        os.path.join(data_dir, "e1rel_to_e2_ranking_test.json")
    )
    write_training_graph(
        all_cases,
        label_graph,
        os.path.join(data_dir, "e1rel_to_e2_full.json")
    )
