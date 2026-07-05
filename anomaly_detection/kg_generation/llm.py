import configparser
import json
import logging
import os
import re
import socket
import time
import unicodedata
import yaml

from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Any, Callable, Iterable, TypeAlias

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from mpi4py import MPI
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch

logger = logging.getLogger(__name__)

@dataclass
class ExtractedEntity:
    text: str
    type: str
    start: int
    end: int

    def span(self) -> tuple[int, int]:
        return (self.start, self.end)

MaskRule: TypeAlias = tuple[re.Pattern, str]

# TODO(lucas): This should be temporary until I replace Drain3 with my own regex masking
def load_masking_rules(ini_path: str) -> list[MaskRule]:
    """
    Load regex patterns and masks from drain3.ini file.
    """
    config = configparser.ConfigParser(interpolation=None)
    config.read(ini_path)

    raw_masking = config["MASKING"]["masking"]

    cleaned_lines = []
    for line in raw_masking.splitlines():
        stripped = line.strip()
        if stripped.startswith(";") or not stripped:
            continue
        cleaned_lines.append(line)

    cleaned_json = "\n".join(cleaned_lines)

    masking_defs = json.loads(cleaned_json)

    compiled_masks: list[MaskRule] = []
    for entry in masking_defs:
        pattern = re.compile(entry["regex_pattern"])
        mask = entry["mask_with"]
        compiled_masks.append((pattern, mask))

    return compiled_masks

def spans_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """
    Spans overlap if each span starts before the other ends.
    """
    result = a[0] < b[1] and b[0] < a[1]
    return result

def extract_regex_entities(log: str, rules: list[MaskRule]) -> list[ExtractedEntity]:
    """
    Extract entities based on regex masking rules.
    """
    entities: list[ExtractedEntity] = []
    for pattern, mask in rules:
        for match in pattern.finditer(log):
            start, end = match.span()
            ee = ExtractedEntity(match.group(0), mask, start, end)
            if not any(spans_overlap(ee.span(), regex_ent.span()) for regex_ent in entities):
                entities.append(ee)

    return sorted(entities, key=lambda x: x.start)

def locate_llm_entities(log: str, llm_entities: list[str]) -> list[ExtractedEntity]:
    """
    Find the spans for entities extracted by the LLM.
    """
    found: list[ExtractedEntity] = []
    for ent in llm_entities:
        for match in re.finditer(rf"\b{re.escape(ent)}\b", log):
            start, end = match.span()
            ee = ExtractedEntity(ent, "", start, end)
            found.append(ee)
    return found

# TODO(lucas): When LLM entities are rejected, log the reason for rejection.
def merge_entities_with_span_guard(
    regex_ents: list[ExtractedEntity],
    llm_ents: list[ExtractedEntity]
) -> list[ExtractedEntity]:
    """
    Add entities found by the LLM only if their spans do not overlap with any regex entities,
    which are more reliable.
    """
    merged: list[ExtractedEntity] = []
    for e in regex_ents:
        merged.append(e)

    # Sort LLM entities from longest to shortest to prevent erroneous substring matches
    llm_ents_sorted = sorted(llm_ents, key=lambda e: e.end - e.start, reverse=True)

    for llm_ent in llm_ents_sorted:
        if not any(spans_overlap(llm_ent.span(), regex_ent.span()) for regex_ent in regex_ents):
            merged.append(llm_ent)

    return merged

"""
Each rule maps (entity_text, regex_type) to iterable[(entity_text, ontology_type)] In other words,
it maps an entity with one regex type to an entity with zero, one, or many ontology types.
Some types have a one-to-one mapping (e.g., <:IP:> -> <:slogert:Address:>), and others
have a one-to-many mapping (e.g., <:EMAIL:> -> <:slogert:User:>@<:slogert:URL:>)
"""
RegexToOntologyRule = Callable[[ExtractedEntity], Iterable[ExtractedEntity]]
ContextualRule = Callable[[ExtractedEntity, str], Iterable[ExtractedEntity]]

def map_ip(entity: ExtractedEntity) -> Iterable[ExtractedEntity]:
    yield ExtractedEntity(entity.text, "slogert:Address", entity.start, entity.end)

def map_timestamp(entity: ExtractedEntity) -> Iterable[ExtractedEntity]:
    yield ExtractedEntity(entity.text, "xsd:dateTime", entity.start, entity.end)

def map_duration(entity: ExtractedEntity) -> Iterable[ExtractedEntity]:
    yield ExtractedEntity(entity.text, "xsd:duration", entity.start, entity.end)

def map_session(entity: ExtractedEntity) -> Iterable[ExtractedEntity]:
    yield ExtractedEntity(entity.text, "slogert:Parameter", entity.start, entity.end)

def map_user(entity: ExtractedEntity) -> Iterable[ExtractedEntity]:
    yield ExtractedEntity(entity.text, "slogert:User", entity.start, entity.end)

def map_url(entity: ExtractedEntity) -> Iterable[ExtractedEntity]:
    yield ExtractedEntity(entity.text, "slogert:URL", entity.start, entity.end)

def map_file_path(entity: ExtractedEntity) -> Iterable[ExtractedEntity]:
    yield ExtractedEntity(entity.text, "slogert:File", entity.start, entity.end)

def map_directory(entity: ExtractedEntity) -> Iterable[ExtractedEntity]:
    yield ExtractedEntity(entity.text, "slogert:File", entity.start, entity.end)

def map_email(entity: ExtractedEntity) -> Iterable[ExtractedEntity]:
    text = entity.text
    start = entity.start
    at_idx = entity.text.find("@")

    if at_idx == -1:
        # This should never happen
        yield entity
        return

    # Split parts
    user = text[:at_idx]
    url = text[at_idx+1:]

    # Get span for each part
    user_start = start
    user_end = start + at_idx
    url_start = start + at_idx + 1
    url_end = start 

    yield ExtractedEntity(user, "slogert:User", user_start, user_end)
    yield ExtractedEntity(url, "slogert:URL", url_start, url_end)

def map_file_name(entity: ExtractedEntity) -> Iterable[ExtractedEntity]:
    yield ExtractedEntity(entity.text, "slogert:File", entity.start, entity.end)

def map_version(entity: ExtractedEntity) -> Iterable[ExtractedEntity]:
    yield ExtractedEntity(entity.text, "slogert:Parameter", entity.start, entity.end)

def map_cmd(entity: ExtractedEntity) -> Iterable[ExtractedEntity]:
    yield ExtractedEntity(entity.text, "slogert:Parameter", entity.start, entity.end)

def map_num(entity: ExtractedEntity) -> Iterable[ExtractedEntity]:
    yield ExtractedEntity(entity.text, "xsd:integer", entity.start, entity.end)

def expand_span(log: str, span: tuple[int, int], pattern: re.Pattern
) -> Iterable[tuple[str, tuple[int, int], re.Match]]:
    """
    Expand an entity span using a regex that must match and include
    the original span.
    Returns: (matched_text, absolute_span, match_object)
    """
    for m in pattern.finditer(log):
        if m.start() <= span[0] and span[1] <= m.end():
            yield m.group(0), m.span(), m

HOST_PATTERN = re.compile(r"[a-zA-z][a-zA-Z0-9\-]*-\d+")
PROC_PATTERN = re.compile(r"([a-zA-z_][a-zA-Z0-9_-]*)\[(\d+)\]")
def map_host(entity: ExtractedEntity, log: str) -> Iterable[ExtractedEntity]:
    """
    Map a hostname that appears, for example, as mail-1. The original regex
    would identify it as mail-<:NUM:>. Expand the span and map it to
    slogert:SourceHost.
    """
    if entity.type != "NUM":
        return

    for text, span, _ in expand_span(log, (entity.start, entity.end), HOST_PATTERN):
        yield ExtractedEntity(text, "slogert:SourceHost", span[0], span[1])

def map_proc(entity: ExtractedEntity, log: str) -> Iterable[ExtractedEntity]:
    """
    A process with a PID is often listed as something like systemd[<:NUM:>].
    Expand the span to get the process name and map it to slogert:Process.
    """
    if entity.type != "NUM":
        return

    for _, _, m in expand_span(log, entity.span(), PROC_PATTERN):
        proc_start, proc_end = m.span(1)
        pid_start, pid_end = m.span(2)

        proc_name = m.group(1)
        pid = m.group(2)

        yield ExtractedEntity(proc_name, "slogert:Process", proc_start, proc_end)
        yield ExtractedEntity(pid, "xsd:integer", pid_start, pid_end)

def map_plain_integer(entity: ExtractedEntity, log: str) -> Iterable[ExtractedEntity]:
    """
    Fallback rule for plain integers that have no surrounding context.
    """
    if entity.type == "NUM":
        yield ExtractedEntity(entity.text, "xsd:integer", entity.start, entity.end)

CONTEXTUAL_RULES: list[ContextualRule] = [
    map_proc,
    map_host,
    map_plain_integer
]

REGEX_RULES: dict[str, RegexToOntologyRule] = {
    "IP":        map_ip,
    "TIMESTAMP": map_timestamp,
    "DURATION":  map_duration,
    "SESSION":   map_session,
    "USER":      map_user,
    "URL":       map_url,
    "FILE_PATH": map_file_path,
    "DIRECTORY": map_directory,
    "EMAIL":     map_email,
    "FILE_NAME": map_file_name,
    "VERSION":   map_version,
    "NUM":       map_num,
    "CMD":       map_cmd
}

def resolve_entities(
    entities: list[ExtractedEntity],
    log: str
) -> tuple[list[ExtractedEntity], list[ExtractedEntity]]:
    """
    Resolve extracted entities into ontology-aligned entities using a
    two-stage rule pipeline.

    This function processes a list of `ExtractedEntity` objects and attempts
    to transform them into semantically meaningful entities with updated
    types and/or spans.

    Process:

    1. Contextual Rules (span-aware):
      - Operate on the full log message and original entity span
      - May extend the span and/or split entities into multiple new ones
      - If a rule produces output, the original entity span is marked as consumed

    2. Regex rules (type-based fallback):
      - Applied only if no contextual rule matches a given entity
      - Operate on the entity text alone
      - Produce one or more normalized entities

    Any entity that cannot be resolved by either stage is added to the
    unresolved list for the LLM to classify.
    """
    resolved: list[ExtractedEntity] = []
    unresolved: list[ExtractedEntity] = []

    # Tracks spans that have already been consumed by a contextual rule
    consumed_spans: set[tuple[int, int]] = set()

    for entity in entities:
        if any(spans_overlap(entity.span(), s) for s in consumed_spans):
            continue

        # Try contextual rules first
        for contextual_rule in CONTEXTUAL_RULES:
            produced_entities = list(contextual_rule(entity, log))
            if produced_entities:
                resolved.extend(produced_entities)
                consumed_spans.add(entity.span())
                for e in produced_entities:
                    consumed_spans.add(e.span())
                break

        # If no contextual rule fits, fall back to regex rules
        else:
            regex_rule = REGEX_RULES.get(entity.type)

            if regex_rule:
                produced_entities = list(regex_rule(entity))
                if produced_entities:
                    resolved.extend(produced_entities)
                else:
                    unresolved.append(entity)

            # If no rule fits, add to unresolved list,
            # which will be fed to the LLM.
            else:
                unresolved.append(entity)

    return resolved, unresolved

def format_seconds(seconds: float) -> str:
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

def get_entity_idx(s: str, entity_list: list[ExtractedEntity]) -> int:
    idx = -1
    for i, entity in enumerate(entity_list):
        if s == entity.text:
            idx = i
            break

    return idx

# TODO(lucas): Have separate variables for the different stages of entity_list instead of mutating it everywhere
def get_entity_list(s: str) -> list[str]:
    # NOTE(lucas): The first item in the entity list could contain some preamble.
    # For instance, some models like to start with a sentence that ends in a boxed environment.
    # In these cases, the entity list should still be identifiable by an opening bracket,
    # an opening parenthesis, or both, and the end needs to be trimmed as well.
    quotes = ["\'", "\""]
    entity_list = filter_chars(s, quotes)

    if not entity_list:
        logging.error("Something went wrong")
        return []

    entity_list = entity_list.splitlines()

    numbered = False
    for item in entity_list:
        # TODO(lucas): This might be better with regex because sometimes the model will have, for example, an IP with 1. in it.
        # It also might have an explanation with a numbered list.
        if "1. (" in item or "1) (" in item:
            numbered = True
            logging.info("Entity list is numbered")
            break

    sep = ""
    if not numbered:
        for item in entity_list:
            if "* " in item:
                sep = "* "
                logging.info("Entity list uses * for list")
                break
            elif "- " in item:
                sep = "- "
                logging.info("Entity list uses - for list")
                break
            elif ", " in item:
                sep = ", "
                logging.info("Entity list uses , for list")
                break

        if sep == "":
            logging.error("No list separator found")

    if numbered is True:
        entity_list = split_numbered_list(s)
    elif sep == ", ":
        entity_str = ""
        # Find the first line with a comma-separated list
        for line in entity_list:
            if re.search(r'\w+\s*,\s*\w+', line):
                entity_str = str(line)
                break

        # If the list is contained inside brackets, get only that text
        m = re.search(r'[\[\{\(].*[\]\}\)]', entity_str)
        if m:
            entity_str = m.group()

        # Remove any brackets around list
        chars_to_remove = ['[', ']', '(', ')', '{', '}']
        entity_str = filter_chars(entity_str, chars_to_remove)

        # Split the entities on the comma, keeping only the last word
        entity_list = [part.strip().split()[-1] for part in re.split(r'\s*,\s*', entity_str) if part.strip()]
    else:
        entity_list = list(filter(lambda s: filter_str(s, sep), entity_list))

    for i, entity in enumerate(entity_list):
        entity_list[i] = entity.replace(sep, "", 1).strip()

        # TODO(lucas): Filtering out "=" at the end gets rid of false entities like "user=",
        # but it might interfere with a Base64 string
        if entity.endswith("=") or entity == "" or entity is None:
            logging.warning(f"Invalid entity: {entity}")
            entity_list.remove(entity)

    return list(dict.fromkeys(entity_list))


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

PAIR_PATTERN = re.compile(
    r"""
    ^\s*
    (?:[-*+]\s*)?   # Optional bullet
    [\(\[]+\s*      # Opening (, [, or [(
    (?P<first>.+?)  # First value
    \s*,\s*
    (?P<second>.+?) # Second value
    \s*[\)\]]+      # Closing ), ], or )]
    \s*,?           # Optional comma
    \s*$
    """,
    re.VERBOSE,
)

# Markdown/narrative format: entity appears in quotes, type must be searched for
QUOTED_ENTITY_PATTERN = re.compile(f"['\"](.+?)['\"]")

def extract_valid_pairs(
    text: str,
    valid_entities: list[ExtractedEntity],
    valid_types: list[str],
) -> list[ExtractedEntity]:
    """
    Extract (entity, type) pairs from LLM output and map them back to
    existing ExtractedEntity objects.

    Only entities that already exist in `valid_entities` are considered valid.
    Types must be in `valid_types`.

    Returns:
        A list of ExtractedEntity objects with updated types.
    """
    def clean_value(value: str) -> str:
        value = value.strip()

        # Strip matching quotes
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {"'", '"'}
        ):
            value = value[1:-1]

        return value.strip()

    results: list[ExtractedEntity] = []

    # Map text -> list of entities
    # This handles entity text that appears in multiple places in the log message
    entity_map: dict[str, list[ExtractedEntity]] = defaultdict(list)
    for e in valid_entities:
        entity_map[e.text].append(e)

    # Track which entities we've already assigned (avoid duplicates)
    assigned: set[int] = set()

    # Pass 1: strict tuple parsing
    for line in text.splitlines():
        match = PAIR_PATTERN.match(line)
        if not match:
            continue

        entity_text = clean_value(match.group("first"))
        entity_type = clean_value(match.group("second"))

        if entity_text not in entity_map:
            logging.error(f": Invalid pair (entity): ({entity_text}, {entity_type})")
            continue
        if entity_type not in valid_types:
            logging.error(f"Invalid pair (type): ({entity_text}, {entity_type})")
            continue

        for ent in entity_map[entity_text]:
            if id(ent) in assigned:
                logging.debug(f"Entity {ent} already exists")
                continue

            ent.type = entity_type
            results.append(ent)
            assigned.add(id(ent))

    # Pass 2: narrative / markdown blocks
    blocks = re.split(r"\n\s*\n", text)

    for block in blocks:
        entity_match = QUOTED_ENTITY_PATTERN.search(block)
        if not entity_match:
            continue

        entity_text = entity_match.group(1)

        if entity_text not in entity_map:
            continue

        found_type = next((t for t in valid_types if t in block), None)

        if not found_type:
            logging.error(f"Could not find valid type for entity '{entity_text}' in block:\n{block}")
            continue

        for ent in entity_map[entity_text]:
            if id(ent) in assigned:
                continue

            ent.type = found_type
            results.append(ent)
            assigned.add(id(ent))

    return results

def parse_triples(text: str) -> list[tuple[str, str, str]]:
    """
    Parse free-text triples into a list of tuples,
    allowing for nested parentheses.
    """
    triples: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    current: list[str] = []
    depth = 0
    token = ""
    
    for c in text:
        if c == "(":
            if depth > 0:
                token += c  # include nested (
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                # End of triple
                current.append(token.strip())

                if len(current) != 3:
                    raise ValueError(
                        f"Expected 3 elements in triple, got {len(current)}: {current}"
                    )

                s, r, o = map(str.strip, current)
                triple = (s, r, o)

                if triple not in seen:
                    seen.add(triple)
                    triples.append(triple)

                current = []
                token = ""
            else:
                token += c
        elif c == "," and depth == 1:
            # Top-level comma separates triple elements
            current.append(token.strip())
            token = ""
        else:
            token += c

    return triples

# TODO(lucas): dataset_rules should probably default to empty str instead of None
def build_ner_prompt(log: str, dataset_rules: str | None = None) -> str:
    prompt = f"""
# Task: Named Entity Recognition

Extract all named entities from the log message.

## Requirements:
- Return ONLY a valid JSON array of strings
- Preserve order
- No explanations or extra text
- Never return prefixes like in=, user=, src=, dst=
- Interpret brackets, parentheses, and separators as delimiters, not as part of entity names.
- Combine date+time when possible
- Treat full file paths as single entities
- Do not consider the word "file" or "files" to be a file itself. Only a file path should be considered a file.

{dataset_rules or ""}

## Log Message:
{log}
"""
    return prompt

def build_classification_prompt(log: str, entities, valid_types: str, dataset_rules: str | None = None) -> str:
    prompt = f"""
# Task: Entity Classification

Classify each entity with exactly ONE type.

## Valid Types:
{valid_types}

## Requirements:
- Output a list of (entity, type) pairs
- One type per entity
- No explanations or extra text

{dataset_rules or ""}

## Log:
{log}

## Entities:
{entities}
"""
    return prompt

def build_triple_prompt(log: str, entities: str, relations: str, dataset_rules: str | None = None) -> str:
    prompt = f"""
# Task: Knowledge Graph Triple Generation

Generate triples using the provided entities.

## Valid Relations:
{relations}

## Requirements:
- NEVER output None, null, or empty values
- Only output triples where BOTH subject and object are entities that appear in the list below
- Use only provided ontology relations
- Maximize coverage when supported by the ontology and log evidence.
- Do NOT invent relations solely to make an entity appear.
- It is acceptable for some entities to have no generated triples.
- Self-referential triples are VALID and sometimes required
- Example valid self-reference: (app123, slogert:app.name, app123)
- Do NOT reject a triple only because subject == object
- Format: (subject, relation, object)
- No explanations

{dataset_rules or ""}

## Log:
{log}

## Entities:
{entities}
"""
    return prompt

def build_messages(task_prompt: str) -> list[dict]:
    system_prompt = """
You are a cybersecurity expert specialized in transforming log data into a knowledge graph.
You will be given a log event and contextual information.

You must:
- Extract as much information as posible
- Remain completely accurate
- Strictly follow output format requirements
- Remain compliant with the given ontology
- Never include explanations or extra information
- Follow all requirements exactly. Failure to do so will result in termination.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_prompt}
    ]

    return messages

@dataclass
class LLMCtx:
    # LLM provider ("huggingface" or "openrouter")
    provider: str = ""

    # LLM model name
    model_name: str = ""

    # huggingface model from AutoModelForCausalLM
    model: Any = None

    # huggingface tokenizer from AutoTokenizer
    tokenizer: Any = None

    # Client for calling models through OpenRouter
    client: Any = None

def llm(ctx: LLMCtx,
        messages: list[dict],
        max_new_tokens: int = 512,
        repetition_penalty: float = 1.1,
        temperature: float = 0.4
) -> str:

    if ctx.provider == "huggingface":
        tokenizer = ctx.tokenizer
        model = ctx.model
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True
        )

        inputs = {k: v.to(model.model.embed_tokens.weight.device)
            for k, v in inputs.items()}

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            do_sample=True
        )

        input_length = inputs["input_ids"].shape[-1]
        response = tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True
        ).strip()

        return response

    elif ctx.provider == "openrouter":
        response = ctx.client.chat.completions.create(
            model=ctx.model_name,
            messages=messages,
            max_tokens=8196,
            temperature=temperature,
        )
        
        message = response.choices[0].message

        logging.info("Reasoning:\n%s", message.reasoning)

        # In some situations (e.g., running out of tokens during reasoning),
        # the final content may be None.
        if message.content:
            return message.content.strip()

        # If the model has reasoning and has not returned, something went wrong.
        if hasattr(message, "reasoning"):
            logging.warning(f"Model exhausted output budget while reasoning or decided to return nothing."            )
            return ""

    raise RuntimeError(f"Unsupported LLM provider: {ctx.provider}")

def mask_entities(log: str, entities: list[ExtractedEntity]):
    # Remove control characters
    log = "".join(ch for ch in log if unicodedata.category(ch)[0] != "C")

    # Sort by start position
    entities = sorted(entities, key=lambda e: e.start)

    result = []
    cursor = 0

    for e in entities:
        # Add text before entity
        result.append(log[cursor:e.start])

        # Add masked entity
        result.append(ensure_brackets(e.type))

        # Move cursor forward
        cursor = e.end

    # Add remaining text
    result.append(log[cursor:])

    return "".join(result)

def generate_next_template(
    log: str,
    drain_template: str,
    valid_types: str,
    valid_rels: str,
    ctx: LLMCtx,
    run_logs: list[dict] | None = None
) -> dict:
    run_log = {
        "ner": {},
        "entity_classification": {},
        "triple_extraction": {},
        "log": {}
    }

    log = log.strip()

    # TODO(lucas): This needs to go up a level so that the regex do not get compiled each time.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    drain_ini_path = os.path.join(current_dir, "config", "default", "drain3.ini")
    mask_rules = load_masking_rules(drain_ini_path)
    regex_entities = extract_regex_entities(log, mask_rules)

    ner_dataset_rules = """
## Ambiguous Entities
- Entities that do not have a direct fit for any valid type should be labeled as slogert:Parameter.
- Examples of entities without direct fits are terminal commands and groups.
    """

    ner_prompt = build_ner_prompt(log, ner_dataset_rules)
    ner_messages = build_messages(ner_prompt)

    logging.info(f"NER Prompt: {ner_prompt}")
    ner_response = llm(ctx, ner_messages)
    logging.info(f"NER Response:\n{ner_response}")
    llm_entity_list = get_entity_list(ner_response)
    logging.info(f"Regex entities:\n{regex_entities}")
    logging.info(f"LLM Entity list:\n{llm_entity_list}")

    llm_entities = locate_llm_entities(log, llm_entity_list)
    entity_list = merge_entities_with_span_guard(regex_entities, llm_entities)

    resolved, unresolved = resolve_entities(regex_entities, log)

    llm_entity_strings = [e.text for e in llm_entities]
    unresolved_strings = [e.text for e in unresolved]
    classification_list = unresolved_strings + llm_entity_strings

    entity_classification_prompt = build_classification_prompt(log, str(classification_list), valid_types)
    classification_messages = build_messages(entity_classification_prompt)

    logging.info(f"Entity Classification Prompt: {entity_classification_prompt}")
    entity_classification_response = llm(ctx, classification_messages)
    logging.info(f"Entity Classification Response:\n{entity_classification_response}\n")

    # TODO(lucas): Protect lists from being None/empty.
    # If they are, just redo the prompt with more tokens?
    valid_type_set = [t for t in valid_types.split("\n") if t]
    llm_entities = extract_valid_pairs(entity_classification_response, unresolved + llm_entities, valid_type_set)
    logging.info(f"LLM entities:\n{llm_entities}\n")
    llm_entities_final = []
    for e_llm in llm_entities:
        if not any(spans_overlap(e_llm.span(), e_reg.span()) for e_reg in resolved):
            llm_entities_final.append(e_llm)

    # Collect final entity list and ensure that they are sorted by order of appearance.
    # Order of appearance is important because triples will refer to entities
    # by an index into this list
    classified_entities = resolved + llm_entities_final
    classified_entities = sorted(classified_entities, key=lambda e: e.start)

    missing_entities = []
    # If any entity types do not exist in the list of valid types,
    # add the corresponding entities to a list to be reclassified
    for entity in classified_entities:
        if entity.type.replace("<:", "").replace(":>", "") not in valid_types:
            logging.error(f"Invalid type for entity: ({entity.text, entity.type})")
            missing_entities.append(entity.text)

    entity_list_text = str([e.text for e in classified_entities])

    # TODO(lucas): This should be imported from a config file to allow for different datasets
    triple_dataset_rules = """
## Entity role constraints
- Some entities are attached to triples later in the process and should not be connected here.
- slogert:SourceHost entities should not participate in generated triples.

## Relation Constraints
- slogert:cmd is self-referential and applied only to slogert:Parameter when it refers to a bash command.
- slogert:app.name is self-referential and applies only to slogert:Application entities.
- slogert:proc.name is self-referential and applies only to slogert:Process entities.
- slogert:proc.id must connect a slogert:Process entity to its xsd:integer identifier and is NOT self-referential.
    """

    triple_extraction_prompt = build_triple_prompt(log, entity_list_text, valid_rels, triple_dataset_rules)
    triple_messages = build_messages(triple_extraction_prompt)
    triple_extraction_response = llm(ctx, triple_messages)

    logging.info(f"Triple Extraction Prompt:\n{triple_extraction_prompt}")
    logging.info(f"Triple Extraction Response:\n{triple_extraction_response}\n")

    triples_out = []
    triples = triple_extraction_response.split('\n')
    # TODO(lucas): Sometimes, the model will generate a full list, then try to correct itself
    # and only manage to generate a partial second list. This pattern can pick up both lists,
    # which results in an error. When this situation occurs, the model usually has multiple
    # newlines and some normal text before the second list.
    triples = parse_triples(triple_extraction_response)
    logging.info("Triples:\n" + "\n".join(f"({t})" for t in triples))

    for triple in triples:
        sub, _, obj = triple
        if sub not in entity_list_text and sub != "":
            missing_entities.append(sub)
        if obj not in entity_list_text and obj != "":
            missing_entities.append(obj)

    # TODO(lucas): Can sometimes get a string index out of range error here
    # triples = [triple for triple in triples if triples[1].strip() in valid_rels]

    # if len(missing_entities) > 0:
    #     logging.info(f"\n\nThere were missing entities\n{missing_entities}")
    #     entity_list = entity_list.union(missing_entities)

    #     entity_classification_prompt = "These are the valid types to consider for the following prompt:\n" + valid_types + \
    #                                 "\nFor context, this is the log message in which they appear:\n" + log + "\n" + user_prompts[1] + "This is the list of entities to classify:\n" + str(missing_entities)
    #     response = llm(model, tokenizer, entity_classification_prompt, max_new_tokens=256)
    #     logging.info(f"Entity Classification Prompt:\n{entity_classification_prompt}")
    #     logging.info(f"Entity Classification Response:\n{response}")

    #     missing_entity_classification_list = get_entity_list(response)
    #     missing_entity_pairs = get_entity_pairs(missing_entity_classification_list)
    #     logging.info(f"Missing entities and types:\n{missing_entity_pairs}")
    #     for pair in missing_entity_pairs:
    #         entity_pairs.append(pair)
    #     entity_pairs = [pair for pair in entity_pairs if pair[1].replace("<:", "").replace(":>", "") in valid_types]
    #     entity_pairs.sort(key=lambda x: len(x[0]), reverse=True)
    #     logging.info(f"Final entity pairs:\n{entity_pairs}")

    # TODO(lucas): If there were any invalid relations, try triple extraction one more time and discard any remaining invalid triples
    # triple_extraction_prompt = "Use this list of entities to perform the following task:\n" + str(entity_list) + \
    #                             "\nThese are the valid relations to consider for the following prompt:\n" + valid_rels + \
    #                             "\n" + user_prompts[3] + "\n" + log

    # TODO(lucas): Scan log line for matches against entity_list (maybe rename to entity_set or something to denote exclusivity).
    # Make a new list containing duplicates, where entities are in order of appearance
    # TODO(lucas): entity_pairs should probably also be a set rather than a list

    # entity_dupe_pattern = re.compile(r"\b(" + "|".join(re.escape(entity) for entity in entity_list) + r")\b")
    # entity_dupe_pattern = re.compile(r"\b(" + "|".join(re.escape(entity.text) for entity in entity_list) + r")\b")
    # entity_list_with_dupes = re.findall(entity_dupe_pattern, log)
    # logging.info(f"Full entity list:\n{entity_list_with_dupes}")

    # NOTE(lucas): Occasionally, the model generates Unicode quotes, so replace those just to be safe
    # quotes = ["\'", "'", "'", "\"", "“", "”", "\＂", "\""]

    # TODO(lucas): Consider fusing this loop over the triples with the one directly after parsing
    for triple in triples:
        sub, rel, obj = triple
        sub_idx = get_entity_idx(sub, classified_entities)
        obj_idx = get_entity_idx(obj, classified_entities)

        not_found = sub_idx == -1 or obj_idx == -1

        if sub_idx == -1:
            logging.error(f"Entity {sub} not found for triple {triple}")
        if obj_idx == -1:
            logging.error(f"Entity {obj} not found for triple {triple}")

        if not_found:
            logging.error(f"Entity list had unfound entities: {classified_entities}")
            continue

        if rel not in valid_rels:
            logging.error(f"Relation {rel} is invalid")

        triples_out.append((sub_idx, rel, obj_idx))

    logging.info("Parsed Triples:\n" + "\n".join(str(t) for t in triples_out))

    masked_log = mask_entities(log, classified_entities)

    drain_template = drain_template.replace('"', '\\"')
    log = log.replace('"', '\\"')
    masked_log = masked_log.replace('"', '\\"')

    logging.info(f"Masked log message:\n{masked_log}\n")

    triples_discarded = len(triples) - len(triples_out)
    if len(triples_out) < len(triples):
        logging.info(f"{triples_discarded} triples were discarded due to entities not being found.")

    regex_entities_dict = [asdict(e) for e in resolved]
    llm_entities_dict = [asdict(e) for e in llm_entities_final]

    result = {
        "log": log,
        "drain_template": drain_template,
        "masked_log": masked_log,
        "triples": triples_out,
        "regex_entities": regex_entities_dict,
        "llm_entities": llm_entities_dict
    }

    if run_logs is not None:
        run_log["ner"]["prompt"] = ner_prompt
        run_log["ner"]["response"] = ner_response
        run_log["ner"]["parsed"] = [asdict(e) for e in entity_list]

        run_log["entity_classification"]["prompt"] = entity_classification_prompt
        run_log["entity_classification"]["response"] = entity_classification_response
        run_log["entity_classification"]["parsed"] = [asdict(e) for e in classified_entities]

        run_log["triple_extraction"]["prompt"] = triple_extraction_prompt
        run_log["triple_extraction"]["response"] = triple_extraction_response
        run_log["triple_extraction"]["parsed"] = triples

        run_log["log"]["log"] = log.strip()
        run_log["log"]["drain_template"] = drain_template
        run_log["log"]["masked_log"] = masked_log

        run_logs.append(run_log)

    return result

def write_templates_to_file(templates: list[dict], out_path: str) -> None:
    def format_row(fields: list[str]) -> str:
        parts = []
        for i, field in enumerate(fields):
            if i < len(fields) - 1:
                part = (field + ", ").ljust(len(field) + 2)
            else:
                part = field
            parts.append(part)

        return indent*4 + "{" + "".join(parts) + "}"

    dir_path = os.path.dirname(out_path)
    os.makedirs(dir_path, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        indent = " "*4
        f.write("{\n")
        f.write(f'{indent}"templates":\n')
        f.write(f"{indent}[\n")
        for template_idx, template in enumerate(templates):
            log = template["log"]
            drain_template = template["drain_template"]
            masked_log = template["masked_log"].rstrip()
            triples = template["triples"]
            logging.info(f'template["triples"]: {triples}')
            f.write(f"{indent*2}{{\n")
            f.write(f'{indent*3}"log": "{log}",\n')
            f.write(f'{indent*3}"drain_template": "{drain_template}",\n')
            f.write(f'{indent*3}"template_mined": "{masked_log}",\n')
            f.write(f'{indent*3}"triples":\n{indent*3}[\n')

            # Triples
            if len(triples) == 0:
                logging.error("Empty triples in template. Discarding.")
            else:
                for triple_idx, triple in enumerate(triples):
                    sub_str = f'"subject": {triple[0]}'
                    rel_str = f'"relation": "{triple[1]}"'
                    obj_str = f'"object": {triple[2]}'

                    formatted_line = format_row([sub_str, rel_str, obj_str])

                    f.write(formatted_line)
                    if triple_idx < len(triples) - 1:
                        f.write(",\n")
            f.write(f"\n{indent*3}],\n")

            # Regex entities
            regex_entities = template["regex_entities"]
            f.write(f'{indent*3}"regex_entities":\n{indent*3}[\n')
            for e_idx, e in enumerate(regex_entities):
                text = f'"text": "{e["text"]}"'
                ent_type = f'"type": "{e["type"]}"'
                start = f'"start": {e["start"]}'
                end = f'"end": {e["end"]}'

                f.write(format_row([text, ent_type, start, end]))
                if e_idx < len(regex_entities) - 1:
                    f.write(",\n")
            f.write(f"\n{indent*3}],\n")

            # LLM entities
            llm_entities = template["llm_entities"]
            f.write(f'{indent*3}"llm_entities":\n{indent*3}[\n')
            for e_idx, e in enumerate(llm_entities):
                text = f'"text": "{e["text"]}"'
                ent_type = f'"type": "{e["type"]}"'
                start = f'"start": {e["start"]}'
                end = f'"end": {e["end"]}'

                f.write(format_row([text, ent_type, start, end]))
                if e_idx < len(llm_entities) - 1:
                    f.write(",\n")
            f.write(f"\n{indent*3}]\n")

            # End Template
            if (template_idx) < len(templates)-1:
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

def template_should_regenerate(template: dict) -> bool:
    result = False

    mask_pattern = re.compile(r"(<:[^>]+:>)")
    drain_matches = re.findall(mask_pattern, template["drain_template"])
    llm_matches = re.findall(mask_pattern, template["masked_log"])
    logging.info(f"Drain masked log matches: {drain_matches}")
    logging.info(f"LLM masked log matches: {llm_matches}")

    if len(llm_matches) == 0:
        logging.error(f"No entities found in LLM masked log")
        result = True

    elif len(llm_matches) < len(drain_matches):
        logging.warning(f"LLM generated fewer entities than Drain")
        result = True

    elif (len(template["triples"]) == 0):
        logging.error("No triples generated")

    return result

def generate_templates(log_path: str, template_path: str, config_path: str, valid_types_path: str,
                       valid_rels_path: str, out_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as config_file:
        cfg = yaml.safe_load(config_file)

    with open(valid_types_path, "r", encoding="utf-8") as types_file:
        valid_types = types_file.read().strip()

    with open(valid_rels_path, "r", encoding="utf-8") as rels_file:
        valid_rels = rels_file.read().strip()

    model_name = cfg["model_id"]
    provider = str.lower(cfg["provider"])
    if provider != "huggingface" and provider != "openrouter":
        raise ValueError(f"Invalid LLM provider: {provider}")

    tokenizer = None
    client = None
    model = None

    ctx = LLMCtx()
    ctx.provider = provider
    ctx.model_name = model_name

    if provider == "huggingface":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("Loading model...")
        model_load_start = time.time()

        # If the model cannot fit on one device, spread it across GPUs
        # and offload the rest to the CPU
        # TODO(lucas): Query the system and use available GPUs
        # and percentages of available memory
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            offload_folder="offload",
            offload_state_dict=True,
            max_memory={
                0: "70GiB",
                1: "70GiB",
                "cpu": "128GiB"
            },
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )
        print(f"Model loaded in {format_seconds(time.time() - model_load_start)}")

        ctx.tokenizer = tokenizer
        ctx.model = model

    elif provider == "openrouter":
        client = OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1"
        )
        ctx.client = client

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

    # TODO(lucas): This is repeating the work of split_between_nodes
    per_node = len(logs) // num_nodes
    rem = len(logs) % num_nodes
    start = node_idx*per_node + min(node_idx, rem)

    # Each node collects data about the run to help debug post-processing methods.
    # For each attempt, the raw and parsed output are collected, and success/failure is noted
    node_runs = []

    log_slice = split_between_nodes(logs, node_idx, num_nodes)
    for i, log in enumerate(log_slice):
        drain_template = json.loads(drain_templates[start + i])["template_mined"]
        logging.info(f"Rank {rank} processing log {i+1}/{len(log_slice)}")
        template = generate_next_template(log, drain_template, valid_types, valid_rels, ctx, node_runs)

        # TODO(lucas): Is it possible to justify some of the entities between Drain and LLM templates?
        # e.g., Drain identified a month and several numbers, but the LLM did not convert to a timestamp?
        attempts = 0
        while template_should_regenerate(template) and attempts < 5:
        # while template_should_regenerate(template):
            logging.info("Regenerating template.")
            template = generate_next_template(log, drain_template, valid_types, valid_rels, ctx)
            attempts += 1

        templates.append(template)

    gathered = comm.gather(templates, root=0)
    gathered_run_logs = comm.gather(node_runs, root=0)
    if rank == 0:
        all_templates = [template for templates in gathered for template in templates]
        run_logs = [log for node in gathered_run_logs for log in node]

        print(f"Templates generated in {format_seconds(time.time() - template_gen_start)}")
        write_templates_to_file(all_templates, out_path)

        if run_logs:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            run_logs_dir = os.path.join(current_dir, "..", "..", "tests", "results")
            if not os.path.exists(run_logs_dir):
                os.mkdir(run_logs_dir)
            with open(os.path.join(run_logs_dir, "run_log.json"), "w", encoding="utf-8") as f:
                json.dump(run_logs, f, indent=4)

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
