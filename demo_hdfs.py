import cProfile
import io
import logging
import pstats
import os
import yaml

from anomaly_detection.kg_generation import hdfs_dataset
from anomaly_detection.kg_generation.llm import find_unique_log_formats, generate_templates
from anomaly_detection.kg_generation.kg_generation import generate_kg
from anomaly_detection.kg_completion.kg_completion import kg_completion

logger = logging.getLogger(__name__)

def main():
    """
    The demo code
    """
    logging.basicConfig(
        filename=os.path.join("results", "HDFS", "log.txt"),
        level=logging.DEBUG,
        format="[%(asctime)s]: %(name)s: %(levelname)s: %(message)s"
    )

    current_dir = os.path.dirname(__file__)
    raw_data_dir = os.path.join(current_dir, "data", "HDFS")
    preprocessed_data_dir = os.path.join(raw_data_dir, "preprocessed")

    llm_config_path = os.path.join(current_dir, "config", "llm_config.yaml")
    valid_types_path = os.path.join(current_dir, "config", "valid_types.json")
    valid_rels_path = os.path.join(current_dir, "config", "valid_rels.json")
    gen_templates_path = os.path.join(current_dir, "results", "HDFS", "templates.json")

    # TODO(lucas): If templates have already been generated, consider using Drain3's persistence and inference mode
    # (inference mode uses template_miner.match(log_line) instead of template_miner.add_log_message(log_line))

    """
    Use a default configuration of the Drain3 log parser to find all unique
    log message formats in a dataset. Writes Drain output to JSON lines file,
    and writes the first example of each message to an output text file.
    The contents of the text file will be used as examples to construct a
    set of templates to extract a list of triples from other messages of
    that format. The Drain output can be used to dial in the default
    configuration to ensure that there are few or no duplicate formats.
    """
    # log_files = [os.path.join(raw_data_dir, "raw", "hdfs.log")]
    # TODO(lucas): Fold this function into generate_templates to avoid confusion over
    # the output paths for this function, which are input to generate_templates
    # find_unique_log_formats(log_files, preprocessed_data_dir)
    unique_logs_path = os.path.join(preprocessed_data_dir, "unique_logs.txt")

    # TODO(lucas): Getting some bad LLM responses for HDFS triple generation. Make sure input is well-structured
    # TODO(lucas): Decide where to save output, and print where output was saved
    generate_templates(unique_logs_path, llm_config_path, valid_types_path, valid_rels_path)

    # hdfs_dataset.extract_dataset(raw_data_dir)

    # pr = cProfile.Profile()
    # pr.enable()
    # generate_kg(raw_data_dir, "HDFS")
    # pr.disable()

    # s = io.StringIO()
    # sortby = pstats.SortKey.TIME
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()

    # profile_path = os.path.join("results", "profile.txt")
    # with open (profile_path, "w+", encoding="utf-8") as f:
    #     f.write(s.getvalue())

    # cfg_path = "config/hdfs.yaml"
    # cfg: dict = {}
    # with open(cfg_path, "r", encoding="utf-8") as cfg_file:
    #     cfg = yaml.safe_load(cfg_file)
    # kg_completion(cfg)

if __name__ == "__main__":
    main()
