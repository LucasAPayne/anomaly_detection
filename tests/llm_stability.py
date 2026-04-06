"""
Runs LLM template generation several times on a couple of datasets.
This test has a few purposes:
1. Detect and report any crashses encountered
2. Evaluate the quality of the LLM output over several iterations
"""

import argparse
import logging
import os
import re
import subprocess
import sys
import time
import traceback

from typing import List, Dict, Optional, Tuple

import orjson

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from anomaly_detection.kg_generation import ait_dataset
from anomaly_detection.kg_generation.llm import find_unique_log_formats, format_seconds

def extract_entities(log: str, masked_log: str) -> List[Optional[str]]:
    timestamp_pattern = r"""(
        (?: # Syslog format e.g., "Feb 29 07:22:12"
            [A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}
        )
        |
        (?: # ISO 8601 e.g., "2025-07-28T14:22:12Z"
            \d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?
        )
        |
        (?: # Alternative: "2025/07/28 14:22:12"
            \d{4}/\d{2}/\d{2}[ T]\d{2}:\d{2}:\d{2}
        )
        |
        (?: # Alternative: "Jul 28 2025 14:22:12"
            [A-Z][a-z]{2}\s+\d{1,2}\s+\d{4}\s+\d{2}:\d{2}:\d{2}
        )
    )"""

    # Specific regex are needed for entities that contain spaces
    # This covers potential cases where multiple entities appear in a row,
    # with no literal strings separating them
    regex_overrides = {"TIMESTAMP": timestamp_pattern}

    placeholder_pattern = re.compile(r"(<:([^>]+):>)")
    placeholders = placeholder_pattern.findall(masked_log)

    pattern = re.compile(r"(<:[^>]+:>)")
    template_parts = re.split(pattern, masked_log)

    # This maps entity type strings to a list of start/end positions
    # of where an entity of the type appears in the original log
    regex_spans: Dict[str, List[Tuple[int,int]]] = {}
    for entity_name, regex_str in regex_overrides.items():
        regex = re.compile(regex_str, re.VERBOSE)
        matches = list(regex.finditer(log))
        regex_spans[entity_name] = [(m.start(), m.end()) for m in matches]

    token_positions = []
    static_tokens = []
    for part in template_parts:
        if not part:
            continue
        if part.startswith("<:") and part.endswith(":>"):
            continue

        token = part.strip()
        if token:
            idx = log.find(token)
            if idx != -1:
                token_positions.append((idx, idx + len(token)))
                static_tokens.append(token)

    entities: List[Optional[str]] = []
    cursor = 0
    for placeholder in placeholders:
        entity_type = placeholder[1]
        if entity_type in regex_spans and regex_spans[entity_type]:
            # First, find all regex spans
            start, end = regex_spans[entity_type].pop(0)
            entities.append(log[start:end].strip())
            cursor = end
        else:
            # Use next literal token span as delimiter
            if token_positions:
                next_token_start, next_token_end = token_positions.pop(0)
                if cursor < next_token_start:
                    entity = log[cursor:next_token_start].strip()

                    for spans in regex_spans.values():
                        for s, e in spans:
                            if cursor <= s and e <= next_token_start:
                                piece = log[s:next_token_start].strip()
                                entity = entity.replace(piece, "").strip()

                    if not entity:
                        continue

                    # Determine the masked log segment corresponding to this entity
                    masked_cursor = masked_log.find(placeholder[0])
                    next_masked_literal_pos = masked_log.find(static_tokens[0]) if static_tokens else len(masked_log)
                    masked_segment = masked_log[masked_cursor:next_masked_literal_pos]

                    # If only one placeholder appears in this segment, treat the whole thing as one entity
                    placeholder_count_in_segment = masked_segment.count("<:")
                    if " " in entity and placeholder_count_in_segment <= 1:
                        entities.append(entity)
                    else:
                        local_entities = entity.replace('"', "").split()
                        entities.extend(local_entities)
                    cursor = next_token_end  # Skip literal
            else:
                # Last entity: take the rest
                entity = log[cursor:].strip()
                if not entity:
                    continue

                masked_cursor = masked_log.find(placeholder[0])
                next_masked_literal_pos = masked_log.find(static_tokens[0]) if static_tokens else len(masked_log)
                masked_segment = masked_log[masked_cursor:next_masked_literal_pos]

                # If only one placeholder appears in this segment, treat the whole thing as one entity
                placeholder_count_in_segment = masked_segment.count("<:")
                if " " in entity and placeholder_count_in_segment <= 1:
                    entities.append(entity)
                else:
                    local_entities = entity.replace('"', "").split()
                    entities.extend(local_entities)
                cursor = len(log)

    print(f"Log: {log}")
    print(f"Masked Log: {masked_log}")
    print(f"Entities: {', '.join(entities)}")
    print("\n")

    return entities

def gen_triples_from_template(triples_template: List[Dict], entities: List[Optional[str]], label: int, log_id: int) -> List[str]:
    def clean_element(el: str) -> str:
        """
        Replace spaces in an element with underscores. Multiple spaces are replaced with
        one underscore, and any trailing spaces are discarded.
        """
        words: list[str] = [word for word in el.lower().split(" ") if word != ""]
        result: str = "_".join(words)
        return result

    invalid_entity_count = 0
    max_idx = len(entities)-1
    triples: List[str] = []
    for template in triples_template:
        if len(entities) == 0:
            print("Entities empty")
            continue
            
        sub_idx = template["subject"]
        obj_idx = template["object"]

        if sub_idx > max_idx or obj_idx > max_idx:
            invalid_entity_count += 1
            print("Subject or object index out of range:")
            print(f"Max: {max_idx}, sub_idx: {sub_idx}, obj_dx: {obj_idx}")
            continue

        sub: str = clean_element(entities[sub_idx])
        rel: str = clean_element(template["relation"])
        obj: str = clean_element(entities[obj_idx] )

        triples.append(f"{sub}\t{rel}\t{obj}\t{label}\t{log_id}")

    if invalid_entity_count > 0:
        print(f"Num invalid entities: {invalid_entity_count}")
        print("\n")

    return triples

# parsed_lines_path is the path to the file containing all the lines of the dataset parsed by Drain
def extract_triples(parsed_lines_path: str, templates_path: str, out_path: str) -> None:
    templates = []
    with open(templates_path, "rb") as templates_file:
        file_contents = templates_file.read()
        templates = orjson.loads(file_contents)["templates"]

    parsed_lines = []
    with open(parsed_lines_path, "rb") as infile:
        for line in infile:
            parsed_lines.append(orjson.loads(line))

    buffer = []
    for parsed_line in parsed_lines:
        for template in templates:
            if parsed_line["template_mined"] == template["drain_template"]:
                llm_template = template["template_mined"]
                # print(template["drain_template"])
                # print(llm_template)
                entities = extract_entities(parsed_line["log"], llm_template)
                label = parsed_line["label"]
                log_id = parsed_line["log_id"]
                triples_template = template["triples"]
                triples = gen_triples_from_template(triples_template, entities, label, log_id)
                buffer.extend(triples)

    with open(out_path, "w", encoding="utf-8") as out_file:
        for triple in buffer:
            out_file.write(triple + "\n")

def run_ait_test(root_dir: str, current_dir: str, iteration: int) -> None:
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
    valid_types_path = os.path.join(root_dir, "config", "valid_types.txt")
    valid_rels_path = os.path.join(root_dir, "config", "valid_rels.txt")
    gen_templates_path = os.path.join(current_dir, "results", "AIT", f"templates_{iteration}.json")

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

def run_hdfs_test(iteration: int) -> None:
    start = time.time()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(current_dir, "..")
    raw_data_dir = os.path.join(root_dir, "data", "HDFS")
    preprocessed_data_dir = os.path.join(raw_data_dir, "preprocessed")
    unique_logs_path = os.path.join(preprocessed_data_dir, "unique_logs.txt")
    unique_templates_path = os.path.join(preprocessed_data_dir, "unique_templates.jsonl")

    if not os.path.exists(unique_logs_path) or not os.path.exists(unique_templates_path):
        log_files = [os.path.join(raw_data_dir, "raw", "hdfs.log")]
        find_unique_log_formats(log_files, preprocessed_data_dir)

    llm_config_path = os.path.join(root_dir, "config", "llm_config.yaml")
    valid_types_path = os.path.join(root_dir, "config", "valid_types.txt")
    valid_rels_path = os.path.join(root_dir, "config", "valid_rels.txt")
    gen_templates_path = os.path.join(current_dir, "results", "HDFS", f"templates_{iteration}.json")

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

    logging.info(f"HDFS run completed in {format_seconds(time.time() - start)}")

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
        description="Run a stability test for LLM template generation for KG generation"
    )
    parser.add_argument("-i", "--iterations", type=int, required=True,
                        help="The number of times to run the test for each dataset")
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(current_dir, "..")

    crash_count = 0
    for i in range(args.iterations):
        print(f"Starting run {i+1} out of {args.iterations}...", flush=True)

        old_crash_count = crash_count

        try:
            logging.info("#### AIT ####")
            run_ait_test(root_dir, current_dir, i)
            # logging.info("#### HDFS ####")
            # run_hdfs_test(i)
        except Exception:
            crash_count += 1
            logging.info(f"Crash count #{crash_count} on iteration {i+1}")
            traceback.print_exc()
        finally:
            if old_crash_count == crash_count:
                logging.info(f"Run {i+1} was successful")

    logging.info(f"Total crashes: {crash_count} out of {args.iterations} runs")

    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # parsed_lines_path = os.path.join(current_dir, "..", "anomaly_detection", "kg_generation", "templates", "AIT", "train.jsonl")
    # templates_path = os.path.join(current_dir, "results", "AIT", "templates_1.json")
    # out_path = os.path.join(current_dir, "results", "AIT", "train.txt")

    # extract_triples(parsed_lines_path, templates_path, out_path)
