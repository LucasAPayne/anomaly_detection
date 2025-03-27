"""
This module generates a KG from raw log files
"""
# TODO(lucas): Give an overview of the process in the docstring above
# TODO(lucas): Add more logging

import fileinput
import logging
import multiprocessing
import multiprocessing.pool
import os
import psutil
import sys
import time

import orjson

from drain3 import TemplateMiner
from drain3.template_miner import ExtractedParameter
from drain3.template_miner_config import TemplateMinerConfig

import numpy as np

entities = set()
relations = set()

# TODO(lucas): Put os wrappers in separate file to be shared
# (For some reason, this function did not work when placed in a separate file)
# TODO(lucas): Give a more precise name like rel_path or relative_path
def join_path(*paths: str) -> str:
    """
    Wrapper around os.path.join to prepend the path of the file from which this function is called

    Parameters
    ----------
    - `*paths`: List of paths to join
    """
    return os.path.join(os.path.dirname(__file__), *paths)

def make_dir(path: str) -> str:
    """
    Wrapper around os.mkdir that creates a directory if it does not exist

    Parameters
    ----------
    - `path`: directory to create
    """
    # NOTE(lucas): Use join_path to make the path relative to the file
    # making the directory
    if not os.path.exists(join_path(path)):
        os.makedirs(join_path(path), exist_ok=True)

    return path

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
    os.remove(path + ".bak")

def parse_log(lines: list[str],
              dataset_name: str,
              logger: logging.Logger = None) -> list[dict]:
    """
    Parse a log file, writing the templates and extracted parameters to a JSON file.

    Parameters
    ----------
    - `in_log_file`: the log file to parse
    - `dataset_name`: the name of the dataset
    - `logger`: optional logger to print progress messages
    """
    config = TemplateMinerConfig()
    config.load(join_path("config", dataset_name, "drain3.ini"))
    config.profiling_enabled = True
    template_miner = TemplateMiner(config=config)

    batch_size = 10_000
    line_count = 0

    start_time = time.time()
    batch_start_time = start_time

    buffer = []

    for i, line in enumerate(lines):
        line = line.rstrip()
        # NOTE(lucas): Temporarily lift label if it comes from test/val set
        # so it does not appear in template. Then put it back
        _, *_, label = line.split()
        line = line.rsplit(None, 1)[0]
        line = line.rstrip()

        # FIXME(lucas): Temporary! HDFS templates need to be updated.
        if dataset_name.lower() == "hdfs":
            line = line.partition(": ")[2]

        result = template_miner.add_log_message(line)
        result["params"] = template_miner.extract_parameters(
                result["template_mined"],
                line,
                exact_matching=True)
        result["label"] = label
        result["log_id"] = i

        line_count += 1
        if line_count % batch_size == 0:
            time_taken = time.time() - batch_start_time
            rate = batch_size / time_taken
            logger.info(f"Processing line: {line_count}, rate {rate:.1f} lines/sec, "
                        f"{len(template_miner.drain.clusters)} clusters so far.")
            batch_start_time = time.time()

        buffer.append(result)

    time_taken = time.time() - start_time
    rate = line_count / time_taken if time_taken > 0 else 0
    logger.info(f"--- Done processing file in {time_taken:.2f} sec. \
        Total of {line_count} lines, rate {rate:.1f} lines/sec, "
        f"{len(template_miner.drain.clusters)} clusters")

    sorted_clusters = sorted(template_miner.drain.clusters, key=lambda it: it.size, reverse=True)
    for cluster in sorted_clusters:
        logger.info(cluster)

    template_miner.profiler.report(0)

    return buffer

def extract_triples(parsed_lines_path: str, out_path: str, dataset_name: str) -> None:
    """
    Extract triples from parsed log files using templates,
    and write the resulting triples to a file.

    Parameters
    ----------
    - `parsed_lines_path`: the file that contains the output of the log parser
    - `out_path`: the file to which to write the extracted triples
    """

    def clean_element(el: str) -> str:
        """
        Replace spaces in an element with underscores. Multiple spaces are replaced with
        one underscore, and any trailing spaces are discarded.
        """
        words: list[str] = [word for word in el.lower().split(" ") if word != ""]
        result: str = "_".join(words)
        return result

    # TODO(lucas): Replace this mapping with mappings to ontologies
    type_map = {"IP": "ip_address",
                "PORT": "port",
                "UID": "user_id",
                "EUID": "effective_user_id",
                "PID": "process_id",
                "AUTH_METHOD": "authentication_method",
                "TIMESTAMP": "timestamp",
                "FILE_PATH": "file_path",
                "EMAIL": "email_address",
                "USER": "user",
                "PROCESS": "process",
                "SUBMODULE": "submodule",
                "SERVICE": "service",
                "MODULE": "module",
                "PROTOCOL": "protocol",
                "SECURITY_STATUS": "security_status",
                "SESSION_ID": "session_id",
                "EVENT": "event",
                "HOST": "host",
                "SESSION": "session",
                "DATANODES": "datanodes",
                "FILEPATH": "filepath",
                "BLOCK": "block"}

    global entities
    global relations

    entities_discarded = 0
    relations_discarded = 0
    triples_discarded = 0

    templates_path = join_path("config", dataset_name, "templates.json")
    templates = []
    with open(templates_path, "rb") as templates_file:
        file_contents = templates_file.read()
        templates = orjson.loads(file_contents)["templates"]

    parsed_lines = []
    with open(parsed_lines_path, "rb") as infile:
        print(parsed_lines_path)
        for line in infile:
            parsed_lines.append(orjson.loads(line))

    print(parsed_lines[0])

    buffer = []
    type_triples = set()

    # TODO(lucas): Try to reduce nesting
    # For each line, search for a matching template
    for parsed_line in parsed_lines:
        for template in templates:
            if template["template_mined"] == parsed_line["template_mined"]:
                # If a matching template is found, get the subject, relation, and object
                # and write a triple to the output file.
                # The template contains the index into the "params" field of the parsed
                # log file
                for relation in template["triples"]:
                    sub_index = relation["subject"]
                    obj_index = relation["object"]
                    sub = clean_element(str(parsed_line["params"][sub_index][0]))
                    obj = clean_element(str(parsed_line["params"][obj_index][0]))
                    rel = clean_element(str(relation["relation"]))
                    label = parsed_line["label"]
                    log_id = parsed_line["log_id"]

                    if "train" in parsed_lines_path:
                        entities.add(sub)
                        relations.add(rel)
                    else:
                        skip = False
                        if sub not in entities or obj not in entities:
                            entities_discarded += 1
                            skip = True
                        if rel not in relations:
                            relations_discarded += 1
                            skip = True
                        if skip:
                            triples_discarded += 1
                            continue

                    buffer.append(f"{sub}\t{rel}\t{obj}\t{label}\t{log_id}\n")

                    # TODO(lucas): Add type relations for objects
                    # Add type relation if subject
                    # sub_type = parse_result["params"][sub_index][1]
                    # obj_type = parse_result["params"][obj_index][1]
                    # if file_in_dataset(infile.name, "train") and sub_type in type_map \
                    #     and sub not in type_triples:
                    #     type_triples.add(sub)
                    #     # TODO(lucas): replace with RDF type relation
                    #     rel = "a"
                    #     obj = type_map[sub_type].lower()
                    #     buffer.append(f"{sub}\t{rel}\t{obj}\t{label}\t{log_id}\n")
                    # if file_in_dataset(infile.name, "train") and obj_type in type_map:
                    #     sub = parse_result["params"][obj_index][0].lower()
                    #     if sub not in type_triples:
                    #         type_triples.add(sub)
                    #         rel = "a"
                    #         obj = type_map[obj_type].lower()
                    #         buffer.append(f"{sub}\t{rel}\t{obj}\t{label}\t{log_id}\n")

    with open(out_path, "w", encoding="utf-8") as out_file:
        out_file.writelines(buffer)

    if triples_discarded:
        print(f"{entities_discarded} entities and {relations_discarded} relations were not found in the train set.")
        print(f"In total, {triples_discarded} triples were discarded.")

def generate_val_set(train_path: str, out_val_path: str, val_ratio: float) -> None:
    """
    Generate a validation set by splitting the train set into two pieces based on val_ratio.
    To make a more varied set, randomly permute the training set.

    Parameters
    ----------
    - `test_path`: path to train set
    - `out_val_path`: path to save validation set to
    - `val_ratio`: percentage of training data to split off for validation data
    """
    with open(train_path, "r", encoding="utf-8") as test_file:
        test_data = np.array(test_file.readlines())

    shuffled_indices = np.random.permutation(len(test_data))
    val_size = int(len(test_data) * val_ratio)
    val_indices = shuffled_indices[:val_size]
    test_indices = shuffled_indices[val_size:]

    with open(train_path, "w", encoding="utf-8") as test_file:
        test_file.writelines(test_data[test_indices])

    with open(out_val_path, "w", encoding="utf-8") as val_file:
        val_file.writelines(test_data[val_indices])

def serialize_default(obj) -> list:
    if isinstance(obj, ExtractedParameter):
        result = [obj.value, obj.mask_name]
        return result

    raise TypeError(f"Type {type(obj)} not serializable")

def extract_triples_multiprocess(extracted_templates: list[dict], train_entities: set,
                                 train_relations: set, dataset_name: str, out_kg_file: str,
                                 num_cores: int) -> None:
    num_templates = len(extracted_templates)
    chunk_size = num_templates // num_cores
    chunks = [extracted_templates[i:i + chunk_size] for i in range(0, num_templates, chunk_size)]

    with multiprocessing.pool.Pool(processes=num_cores) as pool:
        args = [(chunk, train_entities, train_relations, dataset_name, True) for chunk in chunks]
        buffer = pool.starmap(extract_relations, args)
        buffer = [item for sublist in buffer for item in sublist]

    with open(out_kg_file, "w", encoding="utf-8") as out_file:
        out_file.writelines(buffer)

# TODO(lucas): Handle datasets that do not fit in memory
def generate_kg(raw_data_dir: str, dataset_name: str) -> None:
    """
    Generate a knowledge graph from a set of log files using entity and relation extraction.

    Parameters
    ----------
    - `raw_data_dir`: path to directory containing raw log data
    - `dataset_name`: name of the dataset being processed
    """
    logger = logging.getLogger(__name__)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')

    # Directory to write extracted templates to for each log file
    template_dir = make_dir(join_path("templates", dataset_name))

    # Write final KG data to a place where the KG completion module can read it
    preprocessed_data_dir = make_dir(join_path("..", "kg_completion", "datasets", dataset_name))

    train_extracted_templates = []
    train_dir = join_path(raw_data_dir, "train")
    train_file = join_path(template_dir, "train.jsonl")
    test_dir = join_path(raw_data_dir, "test")
    test_file = join_path(template_dir, "test.jsonl")

    # Get only the number of physical cores (no hyperthreading)
    num_cores = psutil.cpu_count(logical=False)
    for root, _, files in os.walk(train_dir):
        for file in files:
            in_log_file = join_path(root, file)
            lines = []
            with open(in_log_file, "r", encoding="utf-8") as infile:
                lines = infile.readlines()

            # Divide lines into as many chunks as there are cores
            num_lines = len(lines)
            chunk_size = num_lines // num_cores
            chunks = [lines[i:i + chunk_size] for i in range(0, num_lines, chunk_size)]

            # Make as many processes as there are cores, and collect all their results
            with multiprocessing.pool.Pool(processes=num_cores) as pool:
                args = [(chunk, dataset_name, logger) for chunk in chunks]
                results = pool.starmap(parse_log, args)

            # Put the results into one continuous list
            templates = [item for sublist in results for item in sublist]
            train_extracted_templates.extend(templates)

    with open(train_file, "wb") as outfile:
        for template in train_extracted_templates:
            outfile.write(orjson.dumps(template, default=serialize_default) + b'\n')

    test_extracted_templates = []
    for root, _, files in os.walk(test_dir):
        for file in files:
            in_log_file = join_path(root, file)
            lines = []
            with open(in_log_file, "r", encoding="utf-8") as infile:
                lines = infile.readlines()

            # Divide lines into as many chunks as there are cores
            num_lines = len(lines)
            chunk_size = num_lines // num_cores
            chunks = [lines[i:i + chunk_size] for i in range(0, num_lines, chunk_size)]

            # Make as many processes as there are cores, and collect all their results
            with multiprocessing.pool.Pool(processes=num_cores) as pool:
                args = [(chunk, dataset_name, logger) for chunk in chunks]
                results = pool.starmap(parse_log, args)

            # Put the results into one continuous list
            templates = [item for sublist in results for item in sublist]
            test_extracted_templates.extend(templates)

    # TODO(lucas): Have option to not save generated template files
    with open(test_file, "wb") as outfile:
        for template in test_extracted_templates:
            outfile.write(orjson.dumps(template, default=serialize_default) + b'\n')

    train_kg_file = os.path.join(preprocessed_data_dir, "train.txt")
    test_kg_file = os.path.join(preprocessed_data_dir, "test.txt")
    val_kg_file = os.path.join(preprocessed_data_dir, "valid.txt")

    train_template_path = join_path(template_dir, "train.jsonl")
    test_template_path = join_path(template_dir, "test.jsonl")
    extract_triples(train_template_path, train_kg_file, dataset_name)
    extract_triples(test_template_path, test_kg_file, dataset_name)

    # remove_duplicate_lines(train_kg_file)
    # remove_duplicate_lines(test_kg_file)
    generate_val_set(test_kg_file, val_kg_file, val_ratio=0.5)
