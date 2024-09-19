"""
Script that extracts a smaller dataset from the AIT log dataset.
The full AIT log dataset can be found here: https://zenodo.org/record/4264796
The intended use for the extracted dataset is knowledge graph completion for anomaly detection.
The extracted dataset is divided into a training and testing set.
The training set contains normal activity.
The testing set contains all data generated from malicious activity.
The testing set also contains some normal activity.
"""

import calendar
import itertools
import multiprocessing
import multiprocessing.pool
import os
import shutil
from datetime import datetime

import psutil
import numpy as np

# TODO(lucas): Put os wrappers in separate file to be shared
# (For some reason, this function did not work when placed in a separate file)
def join_path(*paths: str) -> str:
    """
    Wrapper around os.path.join to prepend this file's path

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

def get_date(log_type: str, line: str) -> str:
    """
    Get timestamp from log line and return date

    Parameters
    ----------
    - `log_type`: type of log file.
      Options are {access, error, audit, exim, suricata, auth, daemon, mail, mail-info,
      messages, sys, user}
    - `line`: line of audit log to be processed
    """
    # AD: abbreviated day
    # AM: abbreviated month
    # d: date (number)
    # m: month (number)

    date = ""

    if log_type == "access":
        # Timestamp format: [d/AM/Y:timestamp]
        timestamp = line[line.find("[")+1:line.find("[")+12]
        day = int(timestamp.split("/")[0])
        month = list(calendar.month_abbr).index(timestamp.split("/")[1])
        year = int(timestamp.split("/")[2])
        date = datetime(year, month, day)
        date = date.strftime("%m/%d/%Y")

    elif log_type == "error":
        # Timestamp format: [AD AM d timestamp Y]
        timestamp = line[line.find("[")+5:line.find("]")]
        month = list(calendar.month_abbr).index(timestamp.split()[0])
        day = int(timestamp.split()[1])
        year = int(timestamp.split()[3])
        date = datetime(year, month, day)
        date = date.strftime("%m/%d/%Y")

    elif log_type == "audit":
        # Timestamp format: msg=audit(timestamp:)
        timestamp = float(line[line.find("audit(")+6:line.find(":")])
        date = datetime.fromtimestamp(timestamp)
        date = date.strftime("%m/%d/%Y")

    elif log_type == "mainlog":
        # Timestamp format: Y-m-d timestamp
        timestamp = line[:line.find(" ")]
        date = datetime.fromisoformat(timestamp)
        date = date.strftime("%m/%d/%Y")

    elif log_type == "fast":
        # Timestamp format: d/m/Y-timestamp
        return line[:line.find("-")]

    elif log_type  in ("auth", "daemon", "mail", "messages", "syslog", "user"):
        # Timestamp format: AM d timestamp
        month = list(calendar.month_abbr).index(line.split()[0])
        day = int(line.split()[1])
        year = 2020
        date = datetime(year, month, day)
        date = date.strftime("%m/%d/%Y")

    return date

def parent_dir(path: str) -> str:
    """
    Return the path to the parent directory of a file

    Parameters
    ----------
    - `path`: path to file
    """
    return path[:path.rfind(os.sep)]

def gather_files(root_dir: str) -> list[str]:
    """
    Gather relevant files recursively from root_dir

    Parameters
    ----------
    - `root_dir`: root directory from which to gather files

    Returns
    ---------
    A list of the file names that were gathered
    """
    file_list = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if not file.startswith(('log', 'attack')) and not file.endswith('json'):
                # Do not include root dir in file path to make it easier to use output paths
                # Also, exclude the first separator ("/", "\", "\\"") to make sure the path
                # does not appear relative
                file_name = os.path.join(root, file)[len(root_dir):]
                if file_name.startswith(os.sep):
                    file_name = file_name[len(os.sep):]
                file_list.append(file_name)

    return file_list

def unzip(zip_path: str, unzipped_name: str) -> None:
    """
    Wrapper around shutil.unpack_archive to include print statements

    Parameters
    ----------
    - `zip_path`: path to zip archive
    - `unzipped_name`: name of extracted file
    """
    if not os.path.exists(unzipped_name) and not zip_path.endswith("log"):
        print(f"Unpacking {zip_path}...", end=' ', flush=True)
        # Extract to same path as zip file (e.g., mail.cup.com/audit)
        shutil.unpack_archive(zip_path, parent_dir(zip_path))
        print("Done")

def remove_unzipped_file(unzipped_path: str) -> None:
    """
    Wrapper around os.remove to include print statements, intended to be used on unzipped files

    Parameters
    ----------
    - `unzipped_path`: path to (unzipped) file
    """
    print(f"Removing unzipped file {unzipped_path}...", end=' ', flush=True)
    os.remove(unzipped_path)
    print("Done")


def extract_archives(file_list: list[str], root_dir: str) -> None:
    """
    Extract zip archives from a file list

    Parameters
    ----------
    - `file_list`: list of files, where the file names do not include the root directory
    - `root_dir`: the root directory of the file list
    """
    for file in file_list:
        # unzip function handles the case where the zip file was already extracted
        if file.endswith("zip") or file.endswith("audit.log"):
            zip_path = join_path(root_dir, file)
            unzipped_name = zip_path.replace("zip", "log")
            unzip(zip_path, unzipped_name)

def exclude_line(line: str, exclusion_list: list[str]) -> bool:
    """
    Determine whether a line should be excluded

    Parameters
    ----------
    - `line`: log line to examine
    - `exclusion_list`: list of words used to exclude log lines
    """
    exclude = False
    for item in exclusion_list:
        if item in line:
            exclude = True
            break
    return exclude

def extract_training_file(file_path: str, exclusion_list: list[str]) -> list[str]:
    buffer = []
    lines = []
    print(f"Extracting training data from {file_path}.")
    with open(file_path, "r", encoding="utf-8") as infile:
        lines = infile.readlines()
        # To find the log type, return the name found between the last slash and the last period
        # If the log file contains the server name (e.g., mail.cup.com-access),
        # remove the last period and everything before
        # Also, remove the "com-" part
        log_type = file_path[file_path.rfind(os.sep)+1:]
        log_type = log_type.replace(".log", "").replace(".info", "")
        log_type = log_type[log_type.rfind(".")+1:]
        log_type = log_type.replace("com-", "")
        valid_dates = ["02/29/2020", "03/01/2020", "03/02/2020", "03/03/2020"]
        for line in lines:
            if get_date(log_type, line) in valid_dates and not \
                exclude_line(line, exclusion_list):
                buffer.append(line.rstrip() + "\t\t0\n")

    return buffer

def extract_training_set(root_dir: str, data_file_list: list, exclusion_list: list[str]) -> None:
    """
    Extract training data
    Attacks occur on 03/04/2020 and 03/05/2020

    Parameters
    ----------
    - `root_dir`: root data directory
    - `data_file_list`: list of all file names to be processed, without the root directory
    - `exclusion_list`: list of words used to exclude log lines
    """
    out_dir = os.path.join(root_dir, "train")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # NOTE(lucas): Each process will work asynchronously on separate files,
    # and they grab another file as soon as their current one is finished.
    num_cores = psutil.cpu_count(logical=False)
    result_path = os.path.join(out_dir, "train.log")
    buffer = []
    with multiprocessing.pool.Pool(processes=num_cores) as pool:
        results = []
        for file in data_file_list:
            if file.endswith("zip"):
                continue

            file_path = os.path.join(root_dir, "data", file)
            results.append(pool.apply_async(extract_training_file,
                                            args=(file_path, exclusion_list)))

        # Flatten the result list
        buffer = list(itertools.chain.from_iterable(result.get() for result in results))

    with open(result_path, "w", encoding="utf-8") as result_file:
        result_file.writelines(buffer)

    print("Training data extracted.")

def extract_testing_file(root_dir: str, file: str, exclusion_list: list[str]) -> list[str]:
    buffer = []

    # NOTE(lucas): Input and output files should be relative to root data directory
    label_file_path = os.path.join(root_dir, "labels", file)
    data_file_path = os.path.join(root_dir, "data", file)

    labels = []
    with open(label_file_path, "r", encoding="utf-8") as label_file:
        labels = label_file.readlines()

    lines = []
    with open(data_file_path, "r", encoding="utf-8") as data_file:
        lines = data_file.readlines()

    print(f"Extracting attack data from {data_file_path}")

    for label_line, data_line in zip(labels, lines):
        if label_line.strip() != "0,0" and not exclude_line(data_line, exclusion_list):
            buffer.append(data_line.rstrip() + "\t\t1\n")

    return buffer

def extract_testing_set(root_dir: str, data_file_list: list[str], exclusion_list: list[str]) -> int:
    """
    Extract all attack data by looping through each line of each file and comparing to the same
    label file
    Attacks occur on 03/04/2020 and 03/05/2020

    Parameters
    ---------
    - `root_dir`: root directory for raw data
    - `data_file_list`: list of all file names to be processed, without the root directory
    - `exclusion_list`: list of words used to exclude log lines
    """
    out_dir = os.path.join(root_dir, "test")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    result_path = os.path.join(out_dir, "test.log")
    num_cores = psutil.cpu_count(logical=False)
    buffer = []
    with multiprocessing.pool.Pool(processes=num_cores) as pool:
        results = []
        for file in data_file_list:
            if file.endswith("zip"):
                continue

            results.append(pool.apply_async(extract_testing_file,
                                            args=(root_dir, file, exclusion_list)))

        buffer = list(itertools.chain.from_iterable(result.get() for result in results))

    with open(result_path, "w", encoding="utf-8") as result_file:
        result_file.writelines(buffer)

    print("Attack data extracted.")
    return len(buffer)

def inject_testing_set(data_dir: str, test_size: int):
    """
    Move some normal data from the train set to the set set so that the test set
    has an equal amount of normal and suspicious data.

    Parameters
    ----------
    - `data_dir`: directory containing raw log file
    - `test_size`: the number of entries in the test set
    """
    print("Injecting training data into test set...", end=' ', flush=True)

    train_path = os.path.join(data_dir, "train", "train.log")
    test_path = os.path.join(data_dir, "test", "test.log")

    with open(train_path, "r+", encoding="utf-8") as train_file, \
         open(test_path, "a", encoding="utf-8") as test_file:
        # NOTE(lucas): Shuffle lines to include data from each original log file
        train_lines = train_file.readlines()
        np.random.shuffle(train_lines)
        test_file.writelines(train_lines[-test_size:])
        train_file.writelines(train_lines[:-test_size])

    print("Done")

def extract_dataset(raw_data_dir: str, exclude_errors: bool=True) -> None:
    """
    Extract a smaller version of the AIT log dataset,
    optionally excluding log lines or zipped files

    Parameters
    ----------
    - `raw_data_dir`: directory containing raw log files
    - `exclude_errors`: exclude log lines containing error/warning/status messages from dataset
    """
    # List of strings to look for to determine whether
    # a log line should be excluded from the dataset
    exclusion_list = []
    if exclude_errors:
        exclusion_list = ["ERROR", "Error", "error", "Status", "Warning", "auditd", "audispd"]

    data_file_list = gather_files(os.path.join(raw_data_dir, "data"))
    # extract_archives(data_file_list, "data/")
    # extract_archives(label_file_list, "labels/")

    extract_training_set(raw_data_dir, data_file_list, exclusion_list)
    test_size = extract_testing_set(raw_data_dir, data_file_list, exclusion_list)
    inject_testing_set(raw_data_dir, test_size)

    # Delete unzipped files
    # for file in data_file_list:
    #     with open(join_path("data", file), encoding="utf-8"):
    #         if file.endswith("audit.log"):
    #             remove_unzipped_file(join_path("data", file))
    #             remove_unzipped_file(join_path("labels", file))
