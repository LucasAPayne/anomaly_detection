"""
Runs LLM template generation several times on a couple of datasets.
This test has a few purposes:
1. Detect and report any crashses encountered
2. Evaluate the quality of the LLM output over several iterations
"""

import argparse
import logging
import os
import time
import traceback

from string import Template

from accelerate import Accelerator

import torch

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from anomaly_detection.kg_generation import ait_dataset
from anomaly_detection.kg_generation.llm import find_unique_log_formats, generate_templates, format_seconds

from accelerate.utils import write_basic_config

logger = logging.getLogger(__name__)

def run_ait_test(iteration: int, accelerator: Accelerator) -> None:
    start = time.time()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(current_dir, "..")
    raw_data_dir = os.path.join(root_dir, "data", "AIT")
    preprocessed_data_dir = os.path.join(raw_data_dir, "preprocessed")
    unique_logs_path = os.path.join(preprocessed_data_dir, "unique_logs.txt")

    if accelerator.is_local_main_process:
        if not os.path.exists(unique_logs_path):
            log_dir = os.path.join(raw_data_dir, "data")
            log_list = ait_dataset.gather_files(log_dir)
            for i, log in enumerate(log_list):
                log_list[i] = os.path.join(log_dir, log)
            find_unique_log_formats(log_list, preprocessed_data_dir)

    accelerator.wait_for_everyone()

    llm_config_path = os.path.join(root_dir, "config", "llm_config.yaml")
    valid_types_path = os.path.join(root_dir, "config", "valid_types.txt")
    valid_rels_path = os.path.join(root_dir, "config", "valid_rels.txt")
    gen_templates_path = os.path.join(current_dir, "results", "AIT", f"templates_{iteration}.json")

    generate_templates(unique_logs_path, llm_config_path, valid_types_path,
                       valid_rels_path, gen_templates_path, accelerator)
    
    if accelerator.is_local_main_process:
        logging.info(f"AIT run completed in {format_seconds(time.time() - start)}")

def run_hdfs_test(iteration: int) -> None:
    start = time.time()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(current_dir, "..")
    raw_data_dir = os.path.join(root_dir, "data", "HDFS")
    preprocessed_data_dir = os.path.join(raw_data_dir, "preprocessed")
    unique_logs_path = os.path.join(preprocessed_data_dir, "unique_logs.txt")

    if not os.path.exists(unique_logs_path):
        log_files = [os.path.join(raw_data_dir, "raw", "hdfs.log")]
        find_unique_log_formats(log_files, preprocessed_data_dir)

    llm_config_path = os.path.join(root_dir, "config", "llm_config.yaml")
    valid_types_path = os.path.join(root_dir, "config", "valid_types.txt")
    valid_rels_path = os.path.join(root_dir, "config", "valid_rels.txt")
    gen_templates_path = os.path.join(current_dir, "results", "HDFS", f"templates_{iteration}.json")

    generate_templates(unique_logs_path, llm_config_path, valid_types_path,
                       valid_rels_path, gen_templates_path)

    logging.info(f"HDFS run completed in {format_seconds(time.time() - start)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a stability test for LLM template generation for KG generation"
    )
    parser.add_argument("-i", "--iterations", type=int, required=True,
                        help="The number of times to run the test for each dataset")
    args = parser.parse_args()

    log_format = "[%(asctime)s]: %(name)s: %(levelname)s: %(message)s"
    file_formatter = logging.Formatter(log_format)

    # TODO(lucas): This is making all logs go to log.txt, when each run should go to a different file.
    logging.basicConfig(
        level=logging.DEBUG,
        format=log_format
    )

    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)

    root_file_handler = logging.FileHandler(os.path.join(log_dir, "log.txt"), mode="w")
    root_file_handler.setLevel(logging.DEBUG)
    root_file_handler.setFormatter(file_formatter)
    logger.addHandler(root_file_handler)

    accelerator = Accelerator()

    crash_count = 0
    for i in range(args.iterations):
        if accelerator.is_local_main_process:
            print(f"Starting run {i+1} out of {args.iterations}...")

        log_path = os.path.join(log_dir, f"run_{i+1}.log")
        file_handler = logging.FileHandler(log_path, mode="w")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        old_crash_count = crash_count

        try:
            if accelerator.is_local_main_process:
                logging.info("#### AIT ####")
            run_ait_test(i, accelerator)
            # logging.info("#### HDFS ####")
            # run_hdfs_test(i)
        except Exception:
            crash_count += 1
            if accelerator.is_local_main_process:
                logging.info(f"Crash count #{crash_count} on iteration {i+1}")
                traceback.print_exc()
        finally:
            if old_crash_count == crash_count and accelerator.is_local_main_process:
                logging.info(f"Run {i+1} was successful")

            logger.removeHandler(file_handler)
            file_handler.close()
    
    if accelerator.is_local_main_process:
        logging.info(f"Total crashes: {crash_count} out of {args.iterations} runs")
