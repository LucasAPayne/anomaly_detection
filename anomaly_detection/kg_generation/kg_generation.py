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

import numpy as np

# NOTE(lucas): global variables
entities = set()
relations = set()

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
    os.remove(path + ".bak")

def parse_log(in_log_file: str,
              out_file: str,
              dataset_name: str,
              batch_size: int = 10000,
              logger: logging.Logger = None,
              labels=True) -> None:
    """
    Parse a log file, writing the templates and extracted parameters to a JSON file.

    Parameters
    ----------
    - `in_log_file`: the log file to parse
    - `out_file`: the file to which to write the results
    - `dataset_name`: the name of the dataset
    - `logger`: optional logger to print progress messages to terminal
    - `label_file`: path to label file if lines include labels
    """

    config = TemplateMinerConfig()
    config.load(join_path("config", dataset_name, "drain3.ini"))
    config.profiling_enabled = True
    template_miner = TemplateMiner(config=config)

    line_count = 0

    print(in_log_file)
    print(out_file)

    with open(in_log_file, "r", encoding="utf-8") as infile, \
        open(out_file, "w", encoding="utf-8") as outfile:
        start_time = time.time()
        batch_start_time = start_time

        buffer = []

        for line in infile:
            line = line.rstrip()
            # NOTE(lucas): Temporarily lift label if it comes from test/val set
            # so it does not appear in template. Then put it back
            label = ""
            # if labels and (file_in_dataset(outfile.name, "test") or \
            #                file_in_dataset(outfile.name, "val")):
            if labels:
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
            # if result["change_type"] != "none":
            #     json.dumps(result)

            buffer.append(json.dumps(result) + "\n")
            if len(buffer) == batch_size:
                outfile.writelines(buffer)
                buffer.clear()

        outfile.writelines(buffer)
        buffer.clear()

    time_taken = time.time() - start_time
    rate = line_count / time_taken if time_taken > 0 else 0
    logger.info(f"--- Done processing file in {time_taken:.2f} sec. \
        Total of {line_count} lines, rate {rate:.1f} lines/sec, "
        f"{len(template_miner.drain.clusters)} clusters")

    sorted_clusters = sorted(template_miner.drain.clusters, key=lambda it: it.size, reverse=True)
    for cluster in sorted_clusters:
        logger.info(cluster)

    template_miner.profiler.report(0)

def extract_relations_templates(template_dir: str, out_path: str, dataset_name: str,
                                labels: bool=True, chunk_size: int=10000) -> None:
    """
    Extract relations from parsed log files using templates,
    and write the resulting triples to a file.

    Parameters
    ----------
    - `template_dir`: the directory containing log template files
    - `out_path`: the file to which to write the extracted triples
    - `labels`: whether log lines contain labels (i.e., number that indicates suspicion).
    """
    # TODO(lucas): Replace this mapping with mappings to ontologies
    type_map = {"IP": "ip_address",
                "PORT": "port",
                "UID": "user_id",
                "PID": "process_id",
                "USER": "user",
                "PROCESS": "process",
                "HOST": "host",
                "SESSION": "session",
                "MODULE": "module",
                "DATANODES": "datanodes",
                "FILEPATH": "filepath",
                "BLOCK": "block"}

    global entities
    global relations

    entities_discarded = 0
    relations_discarded = 0
    triples_discarded = 0

    template_path = join_path("config", dataset_name, "templates.json")
    # TODO(lucas): Try to reduce nesting
    with open(out_path, "w", encoding="utf-8") as out_file, \
         open(template_path, "r", encoding="utf-8") as template_file:
        templates = json.load(template_file)["templates"]

        for entry in os.listdir(template_dir):
            with open(join_path(template_dir, entry), "r", encoding="utf-8") as infile:
                # For each line, search for a matching template

                buffer = []
                for parsed_line in infile:
                    for template in templates:
                        parse_result = json.loads(parsed_line)
                        if template["template_mined"] == parse_result["template_mined"]:
                            # If a matching template is found, get the subject, relation, and object
                            # and write a triple to the output file.
                            # The template contains the index into the "params" field of the parsed
                            # log file
                            for relation in template["triples"]:
                                sub_index = relation["subject"]
                                obj_index = relation["object"]
                                sub = str(parse_result["params"][sub_index][0]).lower()
                                obj = str(parse_result["params"][obj_index][0]).lower()
                                rel = str(relation["relation"]).lower()
                                label = parse_result["label"]

                                if template_dir.endswith("train"):
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

                                if labels:
                                    buffer.append(f"{sub}\t{rel}\t{obj}\t{label}\n")
                                else:
                                    buffer.append(f"{sub}\t{rel}\t{obj}\n")

                                # TODO(lucas): Add type relations for objects
                                # Add type relation if subject
                                sub_type = parse_result["params"][sub_index][1]
                                obj_type = parse_result["params"][obj_index][1]
                                if file_in_dataset(infile.name, "train") and sub_type in type_map:
                                    # TODO(lucas): replace with RDF type relation
                                    rel = "a"
                                    obj = type_map[sub_type].lower()
                                    if labels:
                                        buffer.append(f"{sub}\t{rel}\t{obj}\t{label}\n")
                                    else:
                                        buffer.append(f"{sub}\t{rel}\t{obj}\n")
                                if file_in_dataset(infile.name, "train") and obj_type in type_map:
                                    sub = parse_result["params"][obj_index][0].lower()
                                    rel = "a"
                                    obj = type_map[obj_type].lower()
                                    if labels:
                                        buffer.append(f"{sub}\t{rel}\t{obj}\t{label}\n")
                                    else:
                                        buffer.append(f"{sub}\t{rel}\t{obj}\n")

                    if len(buffer) == chunk_size:
                        out_file.writelines(buffer)
                        buffer.clear()

                out_file.writelines(buffer)
                buffer.clear()

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

def _save_ids(path: str, mapping: dict, chunk_size: int=10000):
    """
    Save mapping of IDs to strings to a file. Internal function

    Parameters
    ----------
    - `path`: location to write the file
    - `mapping`: dictionary containing mappings of entities or relations to IDs 
    """
    # Sort the dictionary by value (outputs list of tuples)
    mapping = sorted(mapping.items(), key=lambda x : x[1])

    with open(path, "w", encoding="utf-8") as outfile:
        buffer = []
        for i in mapping:
            buffer.append(str(i[1]) + '\t' + str(i[0]) + '\n')
            if len(buffer) == chunk_size:
                outfile.writelines(buffer)
                buffer.clear()
        
        outfile.writelines(buffer)
        buffer.clear()

def _regenerate_triples_with_ids(triples_path: str, ent_ids: dict, rel_ids: dict,
                                 labels: bool=True, str=None, chunk_size: int=10000):
    """
    Regenerate triples in a dataset using saved ID mappings.

    Parameters
    ----------
    - `triples_path`: path to triples to regenerate (i.e., train/test/val set)
    - `ent_ids`: mapping of entity strings to IDs
    - `rel_ids`: mapping of relation strings to IDs
    """
    # Check that triples_path is saved as a TTL file
    # If not, rename it to a TTL file
    triples_path_root, triples_path_ext = os.path.splitext(triples_path)
    new_triples_path = triples_path
    if triples_path_ext != ".ttl":
        new_triples_path = triples_path_root + ".ttl"
        os.rename(triples_path, new_triples_path)

    # NOTE(lucas): Create a new file with a different extension for output.
    # Take each line from the input file and write it to the output file, looking up strings in the
    # entity/relation mappings and replacing them with IDs in the output file.
    # If an entity or relation does not exist in the mapping, discard the triple.
    discarded_tripes = 0
    out_path = triples_path_root + ".txt"
    with open(new_triples_path, "r", encoding="utf-8") as infile, \
         open(out_path, "w", encoding="utf-8") as outfile:
        buffer = []
        for line in infile:
            line = line.rstrip().split('\t')

            if line[0] not in ent_ids:
                print(f"{line[0]} does not exist in entity mapping. Discarding triple.")
                discarded_tripes += 1
                continue
            if line[1] not in rel_ids:
                print(f"{line[1]} does not exist in relation mapping. Discarding triple.")
                discarded_tripes += 1
                continue
            if line[2] not in ent_ids:
                print(f"{line[2]} does not exist in entity mapping. Discarding triple.")
                discarded_tripes += 1
                continue

            # All mappings exist
            sub = str(ent_ids[line[0]])
            rel = str(rel_ids[line[1]])
            obj = str(ent_ids[line[2]])

            if labels:
                label = line[3]
                buffer.append(f"{sub}\t{rel}\t{obj}\t{label}\n")
            else:
                buffer.append(f"{sub}\t{rel}\t{obj}\n")

            if len(buffer) == chunk_size:
                outfile.writelines(buffer)
                buffer.clear()

        outfile.writelines(buffer)
        buffer.clear()

    triples_filename = triples_path_root[triples_path_root.rfind(os.sep)+1:]
    print(f"{discarded_tripes} triples discarded from {triples_filename}")

def _generate_ids(preprpocessed_data_dir: str, train_path: str, test_path: str, val_path: str,
                  labels: bool=True) -> None:
    """
    Generate a mapping of entities/relations to IDs based on the training dataset. Then, rebuild
    the KG using those IDs rather than the entities/relations themselves. This can be helpful to
    reduce the size on disk of a dataset, and can also lead to better performance, depending on the
    dataset.

    Parameters
    ----------
    - `preprocessed_data_dir`: directory containing preprocessed data
    - `train_path`: path to preprocessed training set
    - `preserve_old`: whether to preserve old string-based dataset
    """
    ent_ids = {}
    rel_ids = {}

    ent_count = 0
    rel_count = 0

    with open(train_path, "r", encoding="utf-8") as infile:
        for line in infile:
            # Skip prefix definitions in TTL files
            if line.startswith("@prefix") or line == '\n':
                continue

            # Strip trailing whitespace/newline from line and split on tabs to get triple elements
            line = line.rstrip()
            triple = line.split('\t')

            # Add each entity/relation to its corresponding dictionary and increment the count
            if triple[0] not in ent_ids:
                ent_ids[triple[0]] = ent_count
                ent_count += 1
            if triple[1] not in rel_ids:
                rel_ids[triple[1]] = rel_count
                rel_count += 1
            if triple[2] not in ent_ids:
                ent_ids[triple[2]] = ent_count
                ent_count += 1

    _save_ids(os.path.join(preprpocessed_data_dir, "entity_ids.txt"), ent_ids)
    _save_ids(os.path.join(preprpocessed_data_dir, "relation_ids.txt"), rel_ids)

    _regenerate_triples_with_ids(train_path, ent_ids, rel_ids)
    _regenerate_triples_with_ids(test_path, ent_ids, rel_ids)
    _regenerate_triples_with_ids(val_path, ent_ids, rel_ids)

def generate_kg(raw_data_dir: str, dataset_name: str, labels: bool=True, gen_ids: bool=False) -> None:
    """
    Generate a knowledge graph from a set of log files using entity and relation extraction.

    Parameters
    ----------
    - `raw_data_dir`: path to directory containing raw log data
    - `labels`: whether the testing data should be labeled
    """
    # TODO(lucas): Think about converting to all lowercase. Names appear as both, so irwin and
    # Irwin are technically two different entities.
    # TODO(lucas): For the AIT dataset, map names to email addresses to be clear that they refer
    # to the same person
    logger = logging.getLogger(__name__)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')

    # Directory to write KG data to
    preprocessed_data_dir = os.path.join(raw_data_dir, "preprocessed")

    make_dir("templates")
    make_dir(os.path.join("templates", dataset_name))
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
                parse_log(os.path.join(root, file), result_file, dataset_name,
                          logger=logger, labels=labels)

    # TODO(lucas): Have option to remove generated template files and templates directory
    train_kg_file = os.path.join(preprocessed_data_dir, "train.txt")
    test_kg_file = os.path.join(preprocessed_data_dir, "test.txt")
    val_kg_file = os.path.join(preprocessed_data_dir, "valid.txt")

    # TODO(lucas): Remove triples from test/val sets that have entities and relations that do not exist in train set
    # Use Sets
    extract_relations_templates(join_path("templates", "train"), train_kg_file,
                                                      dataset_name, labels=labels)
    extract_relations_templates(join_path("templates", "test"), test_kg_file,
                                dataset_name, labels=labels)
    # remove_duplicate_lines(train_kg_file)
    # remove_duplicate_lines(test_kg_file)
    generate_val_set(test_kg_file, val_kg_file, val_ratio=0.5)

    # Optionally generate ID mappings for each entity and relation and regenerate the triple sets
    # using those IDs
    if gen_ids:
        _generate_ids(preprocessed_data_dir, train_kg_file, test_kg_file, val_kg_file)
