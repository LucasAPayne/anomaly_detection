import json
import logging
import os
import re
import socket
import time
import yaml

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from mpi4py import MPI
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch

logger = logging.getLogger(__name__)

def format_seconds(seconds: int) -> str:
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02}h{minutes:02}m{seconds:02}s"

def filter_chars(input_str: str, chars_to_remove: list[str]) -> str:
    filtered_str = "".join(char for char in input_str if char not in chars_to_remove)
    return filtered_str

def find_unique_log_formats(filenames: list[str], out_dir: str) -> None:
    config = TemplateMinerConfig()
    config.load(os.path.join("anomaly_detection", "kg_generation", "config", "default", "drain3.ini"))
    template_miner = TemplateMiner(config=config)

    out_file = os.path.join(out_dir, "unique_logs.txt")
    template_out_file = os.path.join(out_dir, "unique_templates.jsonl")

    # TODO(lucas): Is there an easy and generic way to filter out archives?
    filenames = list(filter(lambda x: x.endswith("zip") == False, filenames))
    total_templates = 0
    buffer = []
    template_buffer = []
    for filename in filenames:
        with open(filename, "r", encoding="utf=8") as f:
            lines = f.readlines()
            for line in lines:
                result = template_miner.add_log_message(line)
                if result["change_type"] != "none":
                    buffer.append(line)
                    template_buffer.append(result)

    # TODO(lucas): TEMPORARY
    buffer = buffer[:50]
    template_buffer = template_buffer[:50]

    with open(out_file, "w", encoding="utf-8") as out:
        out.writelines(buffer)

    with open(template_out_file, "w", encoding="utf-8") as out:
            for template in template_buffer:
                out.write(json.dumps(template) + "\n")

    logger.info(f"Total unique log message formats found: {len(template_buffer)}")
    logger.info(f"Unique templates were written to {template_out_file}")
    logger.info(f"Original logs were written to {out_file}")

    print(f"Total unique log message formats found: {len(template_buffer)}")
    print(f"Unique templates were written to {template_out_file}")
    print(f"Original logs were written to {out_file}")

def filter_str(s: str, search: str) -> bool:
    keep = True
    if not s or search not in s:
        keep = False

    return keep

def filter_empty_str(s: str) -> bool:
    keep = True
    if not s or s.isspace():
        keep = False

    return keep

def split_numbered_list(s: str) -> list[str]:
    pattern = r"\d+\.\s+|\d+\)\s+"
    matches = list(re.finditer(pattern, s))
    start_index = matches[0].start()
    end_index = matches[-1].end()
    s = s[start_index:end_index]
    parts = re.split(pattern, s)
    result = [part.strip() for part in parts if part.strip()]
    result[-1] = result[-1].split("\n", 1)[0]

    return result

def rreplace(s: str, old: str, new: str, max_split: int=1) -> str:
    li = s.rsplit(old, max_split)
    return new.join(li)

def get_entity_idx(s: str, entity_list: list[str]) -> int:
    idx = -1
    for i, entity in enumerate(entity_list):
        if s == entity:
            idx = i
            break

    return idx

def get_entity_list(s: str) -> set[str]:
    # NOTE(lucas): The first item in the entity list could contain some preamble.
    # For instance, some models like to start with a sentence that ends in a boxed environment.
    # In these cases, the entity list should still be identifiable by an opening bracket,
    # an opening parenthesis, or both, and the end needs to be trimmed as well.
    quotes = ["\'", "\""]
    entity_list = filter_chars(s, quotes).split("\n")

    entity_list[0] = entity_list[0].replace("{", "", 1).replace("[", "", 1).replace("(", "", 1)
    entity_list[-1] = entity_list[-1].replace("}", "", 1).replace("]", "", 1).replace(")", "", 1)

    numbered = False
    for item in entity_list:
        # TODO(lucas): This might be better with regex because sometimes the model will have, for example, an IP with 1. in it.
        # It also might have an explanation with a numbered list.
        if "1. (" in item or "1) (" in item:
            numbered = True
            break

    sep = ""
    for item in entity_list:
        if "*" in item:
            sep = "*"
            break
        elif "- " in entity_list:
            sep = "- "
            break
        elif "," in item:
            sep = ","
            break

    if numbered is True:
        entity_list = split_numbered_list(s)
    else:
        entity_list = list(filter(lambda s: filter_str(s, sep), entity_list))
    for i, s in enumerate(entity_list):
        entity_list[i] = s.replace(sep, "", 1).strip()

        # TODO(lucas): Filtering out "=" at the end gets rid of false entities like "user=",
        # but it might interfere with a Base64 string
        if s.endswith("=") or s == "" or s is None:
            logging.warning(f"Invalid entity: {s}")
            entity_list.remove(s)

    return set(entity_list)

def ensure_brackets(s: str) -> str:
    if not s.startswith("<:"):
        if s.startswith(":"):
            s = "<" + s
        elif s.startswith("<"):
            s = "<:" + s[1:]
        else:
            s = "<:" + s

    if not s.endswith(":>"):
        if s.endswith(":"):
            s = s + ">"
        elif s.endswith(">"):
            s = s[:-1] + ":>"
        else:
            s = s + ":>"

    return s

def get_entity_pairs(classified_entity_list: set[str]) -> list[list[str]]:
    entity_pairs = []
    for item in classified_entity_list:
        pair = filter_chars(item, ["(", ")"]).replace(":]", ":>").split(",")[:2]
        pair[0] = pair[0].strip()
        if len(pair) == 1:
            logging.warning(f"Invalid pair: {pair}")
            continue

        pair[1] = ensure_brackets(pair[1].strip())

        # TODO(lucas): This is to prevent cases where sentences are picked up.
        # Just checking for a space in the first element will invalidate things like timestamps.
        if (" " in pair[0] and "." in pair[0]) or " " in pair[1]:
            logging.warning(f"Inavlid pair: {pair}")
            continue
        entity_pairs.append(pair)
    entity_pairs.sort(key=lambda x: len(x[0]), reverse=True)

    return entity_pairs

def llm(model, tokenizer, prompt: str, max_new_tokens: int=128, rep_penaly: float=1.2, temp: float=0.2) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=rep_penaly,
        temperature=temp)
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return response

def generate_next_template(log: str, drain_template: str, cfg: dict, valid_types: str, valid_rels: str,
                           model, tokenizer) -> dict:
    user_prompts = cfg["prompts"]
    # prompts = [user_prompts[0] + "\n" + log,
    #         "\nThese are the valid types to consider for the following prompt:\n" + valid_types +
    #         "\n" + user_prompts[1],
    #         user_prompts[2] + "\n" + log,
    #         user_prompts[3] + "\n" + log]

    ner_prompt = user_prompts[0] + "\n" + log

    logging.info(f"NER Prompt: {ner_prompt}")
    response = llm(model, tokenizer, ner_prompt)
    logging.info(f"NER Response:\n{response}")
    entity_list = get_entity_list(response)
    logging.info(f"Entity list:\n{entity_list}")

    entity_classification_prompt = "These are the valid types to consider for the following prompt:\n" + valid_types + \
                                "\nFor context, this is the log message in which the entities appear:\n" + log + "\n" + user_prompts[1] + "This is the list of entities to classify:\n" + str(entity_list)
    response = llm(model, tokenizer, entity_classification_prompt, max_new_tokens=256)
    logging.info(f"Entity Classification Response:\n{response}\n")

    # TODO(lucas): Protect lists from being None/empty.
    #  If they are, just redo the prompt with more tokens?
    classified_entity_list = get_entity_list(response)
    entity_pairs = get_entity_pairs(classified_entity_list)

    missing_entities = set()
    # If any entity types do not exist in the list of valid types,
    # add the corresponding entities to a list to be reclassified
    for pair in entity_pairs:
        if pair[1] not in valid_types:
            missing_entities.add(pair[0])

    logging.info(f"Unprocessed classified entities:\n{classified_entity_list}")
    logging.info(f"Classified entities:\n{entity_pairs}\n")

    triple_extraction_prompt = "Use this list of entities to perform the following task:\n" + str(entity_list) + \
                                "\nThese are the valid relations to consider for the following prompt:\n" + valid_rels + \
                                "\n" + user_prompts[3] + "\n" + log
    response = llm(model, tokenizer, triple_extraction_prompt, max_new_tokens=256)
    logging.info(f"Triple Extraction Prompt:\n{triple_extraction_prompt}")
    logging.info(f"Triple Extraction Response:\n{response}\n")

    triples_out = []
    triples = response.split('\n')
    # TODO(lucas): Sometimes, the model will generate a full list, then try to correct itself
    # and only manage to generate a partial second list. This pattern can pick up both lists,
    # which results in an error. When this situation occurs, the model usually has multiple
    # newlines and some normal text before the second list.
    triples = re.findall(r"\(([^)]+)\)", response)
    logging.info("Triples:" + "\n".join(f"({t})" for t in triples))

    for triple in triples:
        try:
            sub, _, obj = filter_chars(triple, ["\'", "\"", "‘", "’"]).split(",")
        except ValueError:
            logging.error(f"Problematic triple (not enough values to split): {triple}")
            continue
        sub = sub.strip()
        obj = obj.strip()
        if sub not in entity_list and sub != "":
            missing_entities.add(sub)
        if obj not in entity_list and obj != "":
            missing_entities.add(obj)

    triples = [triple for triple in triples if triple[1].strip() in valid_rels]

    if len(missing_entities) > 0:
        logging.info(f"\n\nThere were missing entities\n{missing_entities}")
        entity_list = entity_list.union(missing_entities)

        entity_classification_prompt = "These are the valid types to consider for the following prompt:\n" + valid_types + \
                                    "\nFor context, this is the log message in which they appear:\n" + log + "\n" + user_prompts[1] + "This is the list of entities to classify:\n" + str(missing_entities)
        response = llm(model, tokenizer, entity_classification_prompt, max_new_tokens=256)
        logging.info(f"Entity Classification Prompt:\n{entity_classification_prompt}")
        logging.info(f"Entity Classification Response:\n{response}")

        missing_entity_classification_list = get_entity_list(response)
        missing_entity_pairs = get_entity_pairs(missing_entity_classification_list)
        logging.info(f"Missing entities and types:\n{missing_entity_pairs}")
        for pair in missing_entity_pairs:
            entity_pairs.append(pair)
        entity_pairs = [pair for pair in entity_pairs if pair[1] in valid_types]
        entity_pairs.sort(key=lambda x: len(x[0]), reverse=True)
        logging.info(f"Final entity pairs:\n{entity_pairs}")

    # TODO(lucas): If there were any invalid relations, try triple extraction one more time and discard any remaining invalid triples
    # triple_extraction_prompt = "Use this list of entities to perform the following task:\n" + str(entity_list) + \
    #                             "\nThese are the valid relations to consider for the following prompt:\n" + valid_rels + \
    #                             "\n" + user_prompts[3] + "\n" + log

    # TODO(lucas): Scan log line for matches against entity_list (maybe rename to entity_set or something to denote exclusivity).
    # Make a new list containing duplicates, where entities are in order of appearance
    # TODO(lucas): entity_pairs should probably also be a set rather than a list

    entity_dupe_pattern = re.compile(r"\b(" + "|".join(re.escape(entity) for entity in entity_list) + r")\b")
    entity_list_with_dupes = re.findall(entity_dupe_pattern, log)
    logging.info(f"Full entity list:\n{entity_list_with_dupes}")

    # NOTE(lucas): Occasionally, the model generates Unicode quotes, so replace those just to be safe
    quotes = ["\'", "'", "'", "\"", "“", "”", "\＂", "\""]

    for triple in triples:
        elements = triple.split(",")
        if len(elements) < 3:
            logging.error(f"Problematic triple (not enough elements): {elements}\n")
            continue
        elements = [el.strip() for el in elements]
        sub = filter_chars(elements[0], quotes)
        rel = filter_chars(elements[1], quotes)
        if len(elements) > 2:
            obj = filter_chars(elements[2], quotes)
        else:
            logging.warning(f"Invalid triple of len {len(elements)}:" + "\n".join(f"{el}" for el in elements))
            continue

        if "equals" in rel:
            continue

        sub_idx = get_entity_idx(sub, entity_list)
        obj_idx = get_entity_idx(obj, entity_list)

        if sub_idx == -1 or obj_idx == -1 or sub_idx == obj_idx:
            continue

        triples_out.append([sub_idx, rel, obj_idx])

    # entity_dict = dict(entity_pairs)

    # NOTE(lucas): Since these are output to JSON, any quotes need to be escaped
    # quote_chars = ["'", '"', "“", "”", "＂"]
    # escape_quotes_pattern = re.compile(r"[" + "".join(re.escape(c) for c in quote_chars) + r"]")
    # masked_log = escape_quotes_pattern.sub(lambda m: "\\" + m.group(0), log)
    masked_log = log.replace('"', '\\"')
    for entity_pair in entity_pairs:
        masked_log = masked_log.replace(entity_pair[0], entity_pair[1])
    
    drain_template = drain_template.replace('"', '\\"')

    # TODO(lucas): Change this back to info when info messages start appearing again
    logging.warning(f"Masked log message:\n{masked_log}\n")

    triples_discarded = len(triples) - len(triples_out)
    if len(triples_out) < len(triples):
        logging.info(f"{triples_discarded} triples were discarded due to entities not being found.")

    result = {"drain_template": drain_template, "masked_log": masked_log, "triples": triples_out}
    return result

def write_templates_to_file(templates: list[dict], out_path: str) -> None:
    def format_row(sub_str, rel_str, obj_str):
        sub_field = (sub_str + ", ").ljust(len(sub_str) + 2)
        rel_field = (rel_str + ", ").ljust(len(rel_str) + 2)
        return indent*4 + "{" + sub_field + rel_field + obj_str + "}"

    dir_path = os.path.dirname(out_path)
    os.makedirs(dir_path, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        indent = " "*4
        f.write("{\n")
        f.write(f"{indent}\"templates\":\n")
        f.write(f"{indent}[\n")
        for template_idx, template in enumerate(templates):
            drain_template = template["drain_template"]
            masked_log = template["masked_log"].rstrip()
            triples = template["triples"]
            logging.info(f"template[\"triples\"]: {triples}")
            f.write(f"{indent*2}{{\n")
            f.write(f"{indent*3}\"drain_template\": \"{drain_template}\",\n")
            f.write(f"{indent*3}\"template_mined\": \"{masked_log}\",\n")
            f.write(f"{indent*3}\"triples\":\n{indent*3}[\n")

            if len(triples) == 0:
                logging.error("Empty triples in template. Discarding.")
                f.write(f"\n{indent*3}]\n{indent*2}}},\n")
                continue

            sub_entries = [f'"subject": {t[0]}' for t in triples]
            rel_entries = [f'"relation": "{t[1]}"' for t in triples]
            max_sub_len = max(len(s) for s in sub_entries)
            max_rel_len = max(len(r) for r in rel_entries)

            for triple_idx, triple in enumerate(triples):
                sub_str = f'"subject": {triple[0]}'
                rel_str = f'"relation": "{triple[1]}"'
                obj_str = f'"object": {triple[2]}'

                formatted_line = format_row(sub_str, rel_str, obj_str)

                f.write(formatted_line)
                if triple_idx < len(triples) - 1:
                    f.write(",\n")

            f.write(f"\n{indent*3}]\n");
            if template_idx < len(templates) - 1:
                f.write(f"{indent*2}}},\n")
            else:
                f.write(f"{indent*2}}}\n")
        f.write(f"{indent}]\n")
        f.write("}\n")

def split_between_nodes(logs: list[str], node_idx: int, num_nodes: int) -> list[str]:
    per_node = len(logs) // num_nodes
    rem = len(logs) % num_nodes
    start = node_idx*per_node + min(node_idx, rem)
    end = start + per_node + (1 if node_idx < rem else 0)
    return logs[start:end]

def generate_templates(log_path: str, template_path: str, config_path: str, valid_types_path: str,
                       valid_rels_path: str, out_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as config_file:
        cfg = yaml.safe_load(config_file)

    with open(valid_types_path, "r", encoding="utf-8") as types_file:
        valid_types = types_file.read()

    with open(valid_rels_path, "r", encoding="utf-8") as rels_file:
        valid_rels = rels_file.read()

    model_name = cfg["model_id"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Loading model...")

    model_load_start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16
    )

    print(f"Model loaded in {format_seconds(time.time() - model_load_start)}")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # NOTE(lucas): Tends to hang without this barrier
    comm.Barrier()

    with open(log_path, "r", encoding="utf-8") as f:
        logs = f.readlines()

    with open(template_path, "r", encoding="utf-8") as f:
        drain_templates = f.readlines()

    template_gen_start = time.time()
    templates = []

    host = socket.gethostname()
    hosts = comm.allgather(host)
    unique_hosts = sorted(set(hosts))
    node_idx = unique_hosts.index(host)
    num_nodes = len(unique_hosts)

    log_slice = split_between_nodes(logs, node_idx, num_nodes)
    for i, log in enumerate(log_slice):
        drain_template = json.loads(drain_templates[i])["template_mined"]
        logging.info(f"Rank {rank} processing log {i+1}/{len(log_slice)}")
        templates.append(generate_next_template(log, drain_template, cfg, valid_types, valid_rels, model, tokenizer))

    gathered = comm.gather(templates, root=0)
    if rank == 0:
        all_templates = [template for templates in gathered for template in templates]

    if rank == 0:
        print(f"Templates generated in {format_seconds(time.time() - template_gen_start)}")
        write_templates_to_file(all_templates, out_path)

    # TODO(lucas): Move this to a test file
    # examples = [
    #     "example", "<example", ":example", "<:example", "example:", "example>", "example:>",
    #     "<example:", "<example:>", ":example:", ":example>", "example:>", "<:example:", "<:example>", "<:example:>",
    # ]

    # for ex in examples:
    #     test = ensure_brackets(ex)
    #     result = test == "<:example:>"
    #     result_str = "Pass" if result is True else "Fail"
    #     print(f"{ex} -> {ensure_brackets(ex)} ({result_str})")

if __name__ == "__main__":
    # TODO(lucas): Add explanation to each argument
    # logger = logging.getLogger(__name__)
    log_format = "[%(asctime)s]: %(name)s: %(levelname)s: %(message)s"
    logging.basicConfig(
        level=logging.DEBUG,
        format=log_format
    )

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--template-path", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--valid-types-path", required=True)
    parser.add_argument("--valid-rels-path", required=True)
    parser.add_argument("--out-path", required=True)

    args = parser.parse_args()
    generate_templates(args.log_path, args.template_path, args.config_path, args.valid_types_path,
                       args.valid_rels_path, args.out_path)
