"""
This module generates a KG from raw log files
"""
# TODO(lucas): Give an overview of the process in the docstring above
# TODO(lucas): Add more logging

import fileinput
import json
import logging
import os
import sys
import time

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig


# TODO(lucas): Put os wrappers in separate file to be shared
# (For some reason, this function did not work when placed in a separate file)
# TODO(lucas); Give a more precise name like rel_path or relative_path
def join_path(*paths: str) -> str:
    """
    Wrapper around os.path.join to prepend the path of the file from which this function is called

    Parameters
    ----------
    - `*paths`: List of paths to join
    """
    return os.path.join(os.path.dirname(__file__), *paths)

def make_dir(path: str) -> None:
    """
    Wrapper around os.mkdir that creates a directory if it does not exist

    Parameters
    ----------
    - `path`: directory to create
    """
    # NOTE(lucas): Use join_path to make the path relative to the file
    # making the directory
    if not os.path.exists(join_path(path)):
        os.mkdir(join_path(path))

def file_in_dataset(path: str, dataset: str):
    """
    Determines whether a file is in the train, test, or val set
    based on its parent directories.

    Parameters
    ----------
    - `path`: path to file
    - `dataset`: must be one of {"train", "test", "val"}
    """
    train_names = ["train", "training"]
    test_names = ["test", "testing"]
    val_names = ["val", "valid", "validation", "validate", "validating"]
    ret = False
    dirs = path.split(os.sep)

    if dataset == "train":
        ret = bool(len(set(train_names).intersection(dirs)) > 0)

    elif dataset == "test":
        ret = bool(len(set(test_names).intersection(dirs)) > 0)

    elif dataset == "val":
        ret = bool(len(set(val_names).intersection(dirs)) > 0)

    return ret

def remove_duplicate_lines(path: str):
    """
    Remove duplicate lines from a file. The lines of the file are hashed to save space when
    dealing with very large files. This comes at the cost of a low chance of collision. If
    collisions occur, the built-in hash function (64-bit output) can be swapped for a larger hash.

    Parameters
    ----------
    - `path`: path to file
    """
    lines_seen = set()
    with fileinput.input(path, inplace=True, backup=".bak", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            hashed_line = hash(line)
            if hashed_line not in lines_seen:
                lines_seen.add(hashed_line)
                # NOTE(lucas): fileinput redirects stdout to file,
                # so print(line) writes the line to the file
                print(line)

    # Successfully removed duplicates and closed files, now delete backup file
    # FIXME(lucas): train.ttl.bak is removed successfully, but test.ttl.bak is not,
    # even though it reports success in both cases
    os.remove(path + ".bak")

def parse_log(in_log_file: str,
              out_file: str,
              batch_size: int = 10000,
              logger: logging.Logger = None,
              labels=False) -> None:
    """
    Parse a log file, writing the templates and extracted parameters to a JSON file.

    Parameters
    ----------
    - `in_log_file`: the log file to parse
    - `out_file`: the file to which to write the results
    - `logger`: optional logger to print progress messages to terminal
    - `labels`: whether log lines contain labels (i.e., number that indicates suspicion).
                Only affects test/val datasets
    """

    config = TemplateMinerConfig()
    config.load(join_path("config", "drain3.ini"))
    config.profiling_enabled = True
    template_miner = TemplateMiner(config=config)

    line_count = 0

    print(in_log_file)
    print(out_file)

    with open(in_log_file, "r", encoding="utf-8") as infile, \
        open(out_file, "w", encoding="utf-8") as outfile:
        start_time = time.time()
        batch_start_time = start_time

        for line in infile:
            line = line.rstrip()
            # NOTE(lucas): Temporarily lift label if it comes from test/val set
            # so it does not appear in template. Then put it back
            label = ""
            if labels and (file_in_dataset(outfile.name, "test") or \
                           file_in_dataset(outfile.name, "val")):
                _, *_, label = line.split()
                line = line.rsplit(None, 1)[0]
            line = line.rstrip()

            # TODO(lucas): See about removing this to preserve timestamp/additional information
            line = line.partition(": ")[2]

            result = template_miner.add_log_message(line)
            result["params"] = template_miner.extract_parameters(
                    result["template_mined"],
                    line,
                    exact_matching=True)
            result["label"] = label

            line_count += 1
            if line_count % batch_size == 0:
                time_taken = time.time() - batch_start_time
                rate = batch_size / time_taken
                logger.info(f"Processing line: {line_count}, rate {rate:.1f} lines/sec, "
                            f"{len(template_miner.drain.clusters)} clusters so far.")
                batch_start_time = time.time()
            if result["change_type"] != "none":
                json.dumps(result)

            outfile.write(json.dumps(result) + "\n")

    time_taken = time.time() - start_time
    rate = line_count / time_taken if time_taken > 0 else 0
    logger.info(f"--- Done processing file in {time_taken:.2f} sec. \
        Total of {line_count} lines, rate {rate:.1f} lines/sec, "
        f"{len(template_miner.drain.clusters)} clusters")

    sorted_clusters = sorted(template_miner.drain.clusters, key=lambda it: it.size, reverse=True)
    for cluster in sorted_clusters:
        logger.info(cluster)

    template_miner.profiler.report(0)

def extract_relations_templates(template_dir: str, out_file: str, labels=False) -> None:
    """
    Extract relations from parsed log files using templates,
    and write the resulting triples to a file.

    Parameters
    ----------
    - `template_dir`: the directory containing log template files
    - `out_file`: the file to which to write the extracted triples
    - `labels`: whether log lines contain labels (i.e., number that indicates suspicion).
                Only affects test/val datasets
    """
    # TODO(lucas): Replace this mapping with mappings to ontologies
    type_map = {"IP": "ip_address",
                "UID": "user_id",
                "PID": "process_id",
                "USER": "user",
                "PROCESS": "process",
                "HOST": "host",
                "SESSION": "session"}

    with open(out_file, "w", encoding="utf-8") as outfile, \
         open (join_path("config", "templates.json"), "r", encoding="utf-8") as template_file:
        templates = json.load(template_file)["templates"]

        for entry in os.listdir(template_dir):
            with open(join_path(template_dir, entry), "r", encoding="utf-8") as infile:
                # For each line, search for a matching template
                for parsed_line in infile:
                    for template in templates:
                        parse_result = json.loads(parsed_line)
                        if template["template_mined"] == parse_result["template_mined"]:
                            # If a matching template is found, get the subject, relation, and object
                            # and write a triple to the output file.
                            # The template contains the index into the "params" field of the parsed
                            # log file
                            for relation in template["relations"]:
                                sub_index = relation["subject"]
                                obj_index = relation["object"]
                                sub = parse_result["params"][sub_index][0]
                                obj = parse_result["params"][obj_index][0]
                                rel = relation["relation_label"]
                                if labels:
                                    label = parse_result["label"]
                                    outfile.write(f"{sub}\t{rel}\t{obj}\t{label}\n")
                                else:
                                    outfile.write(f"{sub}\t{rel}\t{obj}\n")

                                # TODO(lucas): Add type relations for objects
                                # Add type relation if subject
                                sub_type = parse_result["params"][sub_index][1]
                                obj_type = parse_result["params"][obj_index][1]
                                if file_in_dataset(infile.name, "train") and  sub_type in type_map:
                                    # TODO(lucas): replace with RDF type relation
                                    rel = "a"
                                    obj = type_map[sub_type]
                                    outfile.write(f"{sub}\t{rel}\t{obj}\n")
                                if file_in_dataset(infile.name, "train") and obj_type in type_map:
                                    sub = parse_result["params"][obj_index][0]
                                    rel = "a"
                                    obj = type_map[obj_type]
                                    outfile.write(f"{sub}\t{rel}\t{obj}\n")

def generate_val_set(test_path, val_path):
    """
    Generate a validation set by splitting the test set in half.
    To make a more varied set, alternate which lines go to testing and which go to validation.

    Parameters
    ----------
    - `test_path`: path to test set
    - `val_path`: path to save validation set to
    """
    with fileinput.input(test_path, inplace=True, backup=".bak", encoding="utf-8") as test_file:
        with open(val_path, "w", encoding="utf-8") as val_file:
            for i, line in enumerate(test_file):
                if i % 2 == 0:
                    print(line.strip())
                else:
                    val_file.write(line)

def gen_kg(raw_data_dir: str, labels: bool=True) -> None:
    """
    Generate a knowledge graph from a set of log files using entity and relation extraction.

    Parameters
    ----------
    - `raw_data_dir`: path to directory containing raw log data
    - `labels`: whether the testing data should be labeled
    """
    logger = logging.getLogger(__name__)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')

    # Directory to write KG data to
    preprocessed_data_dir = os.path.join(raw_data_dir, "preprocessed")

    make_dir("templates")
    make_dir(preprocessed_data_dir)

    for root, _, files in os.walk(raw_data_dir):
        for file in files:
            filename = os.path.splitext(file)[0]

            # TODO(lucas): Is there a better way to keep these files separate?
            # TODO(lucas): Map the matches below to train/test/val to be more consistent
            # after this step
            matches = ["train", "training", "test", "testing",
                       "val", "valid", "validation", "validate"]
            match = next((x for x in matches if x in root), False)
            if match:
                result_prefix = root[root.find(match)+len(match)+1:]
                result_prefix = result_prefix.replace("\\", "_") \
                                             .replace("/", "_") \
                                             .replace("\\\\", "_") + "_"
                result_prefix = os.path.join(match, result_prefix)

                make_dir(join_path("templates", match))

                result_file = join_path("templates", result_prefix + filename + "_result.jsonl")
                parse_log(os.path.join(root, file), result_file, logger=logger, labels=labels)

    # TODO(lucas): Have option to remove generated template files and templates directory
    train_kg_file = os.path.join(preprocessed_data_dir, "train.ttl")
    test_kg_file = os.path.join(preprocessed_data_dir, "test.ttl")
    extract_relations_templates(join_path("templates", "train"), train_kg_file)
    extract_relations_templates(join_path("templates", "test"), test_kg_file, labels=True)
    remove_duplicate_lines(train_kg_file)
    remove_duplicate_lines(test_kg_file)
    generate_val_set(join_path(preprocessed_data_dir, "test.ttl"),
                     join_path(preprocessed_data_dir, "val.ttl"))
