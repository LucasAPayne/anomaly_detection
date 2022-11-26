"""
Script that extracts a smaller dataset from the AIT log dataset.
The full AIT log dataset can be found here: https://zenodo.org/record/4264796
The intended use for the extracted dataset is knowledge graph completion for anomaly detection.
The extracted dataset is divided into a training and testing set.
The training set contains a day worth of normal activity.
The testing set contains all data generated from malicious activity.
The testing set also contains some normal activity.
"""

import argparse
import calendar
import os
import shutil
from datetime import datetime


def get_date(log_type: str, line: str) -> str:
    """
    Get timestamp from log line and return date

    Parameters
    ----------
    - log_type: type of log file.
      Options are {access, error, audit, exim, suricata, auth, daemon, mail, mail-info,
      messages, sys, user}
    - line: line of audit log to be processed
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
    - path: path to file
    """
    return path[:path.rfind(os.sep)]

def gather_files(root_dir: str) -> list[str]:
    """
    Gather relevant files recursively from root_dir

    Parameters
    ----------
    - root_dir: root directory from which to gather files

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
    - zip_path: path to zip archive
    - unzipped_name: name of extracted file
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
    - unzipped_path: path to (unzipped) file
    """
    print(f"Removing unzipped file {unzipped_path}...", end=' ', flush=True)
    os.remove(unzipped_path)
    print("Done")


def extract_archives(file_list: list[str], root_dir: str) -> None:
    """
    Extract zip archives from a file list

    Parameters
    ----------
    - file_list: list of files, where the file names do not include the root directory
    - root_dir: the root directory of the file list
    """
    for file in file_list:
        # unzip function handles the case where the zip file was already extracted
        if file.endswith("zip") or file.endswith("audit.log"):
            zip_path = os.path.join(root_dir, file)
            unzipped_name = zip_path.replace("zip", "log")
            unzip(zip_path, unzipped_name)

def exclude_line(line: str, exclusion_list: list[str]) -> bool:
    """
    Determine whether a line should be excluded

    Parameters
    ----------
    - line: log line to examine
    - exclusion_list: list of words used to exclude log lines
    """
    exclude = False
    for item in exclusion_list:
        if item in line:
            exclude = True
            break
    return exclude

def extract_training_set(data_file_list: list, exclusion_list: list[str]) -> None:
    """
    Extract training over a 1-day period
    Attacks occur on 03/04/2020 and 03/05/2020

    Parameters
    ----------
    - data_file_list: list of all file names to be processed, without the root directory
    - exclusion_list: list of words used to exclude log lines
    """
    for file in data_file_list:
        if file.endswith("zip"):
            continue

        data_file_to_open = os.path.join("data", file)
        out_file_to_open = os.path.join("output", "train", file)
        # If there is not already a folder for output, create one
        # (make sure not to make the target output file into a directory)
        if not os.path.exists(parent_dir(out_file_to_open)):
            os.makedirs(parent_dir(out_file_to_open))

        with open(data_file_to_open, "r", encoding="utf-8") as in_file, \
             open(out_file_to_open, "w", encoding="utf-8") as out_file:
            print(f"Extracting training data from {file}...", end=' ', flush=True)

            # To find the log type, return the name found between the last slash and the last period
            # If the log file contains the server name (e.g., mail.cup.com-access),
            # remove the last period and everything before
            # Also, remove the "com-" part
            log_type = file[file.rfind(os.sep)+1:]
            log_type = log_type.replace(".log", "").replace(".info", "")
            log_type = log_type[log_type.rfind(".")+1:]
            log_type = log_type.replace("com-", "")
            for line in in_file:
                if get_date(log_type, line) == "03/03/2020" and not \
                   exclude_line(line, exclusion_list):
                    out_file.write(line)

            print("Done")

    print("Training data extracted")


def extract_testing_set(data_file_list: list[str], exclusion_list: list[str]) ->None:
    """
    Extract all attack data by looping through each line of each file and comparing to the same
    label file
    Attacks occur on 03/04/2020 and 03/05/2020

    Parameters
    ---------
    - data_file_list: list of all file names to be processed, without the root directory
    """
    for file in data_file_list:
        if file.endswith("zip"):
            continue

        label_file_to_open = os.path.join("labels", file)
        data_file_to_open = os.path.join("data", file)
        out_file_to_open = os.path.join("output", "test", file)
        # If there is not already a folder for output, create one
        # (make sure not to make the target output file into a directory)
        if not os.path.exists(parent_dir(out_file_to_open)):
            os.makedirs(parent_dir(out_file_to_open))

        with open(label_file_to_open, "r", encoding="utf-8") as label_file, \
             open(data_file_to_open, "r", encoding="utf-8") as data_file, \
             open(out_file_to_open, "w", encoding="utf-8") as out_file:
            print(f"Extracting attack data from {file}...", end=' ', flush=True)

            for label_line, data_line in zip(label_file, data_file):
                if label_line.strip() != "0,0" and not exclude_line(data_line, exclusion_list):
                    out_file.write(data_line.rstrip() + "\t\t0\n")
            print("Done")

    print("Attack data extracted")


def inject_testing_set(data_file_list: list[str], lines: int):
    """
    Put the last few lines of each file in the training set into the testing set

    Parameters
    ----------
    - data_file_list: list of all file names to be processed, without the root directory
    - lines: the number of lines to copy from each training file to each testing file
    """
    print("Injecting training data into test set...", end=' ', flush=True)
    for file in data_file_list:
        if file.endswith("zip"):
            continue

        train_file_to_open = os.path.join("output", "train", file)
        test_file_to_open = os.path.join("output", "test", file)

        with open(train_file_to_open, "r+", encoding="utf-8") as in_file, \
             open(test_file_to_open, "a", encoding="utf-8") as out_file:
            train_lines = in_file.readlines()

            # Append an "observed during training" label to each line from the training set
            lines_to_write = train_lines[-lines:]
            for line in lines_to_write:
                out_file.write(line.rstrip() + "\t\t4\n")

            # Delete the last n lines from training file to prevent duplication
            in_file.writelines(train_lines[:-lines])

    print("Done")


def main():
    """
    Extract a smaller version of the AIT log dataset,
    optionally excluding log lines or zipped files
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--exclude_errors", "-e", action="store_true",
        help="Exclude log line containing error/warning/status messages from dataset")
    args = parser.parse_args()

    # List of strings to look for to determine whether
    # a log line should be excluded from the dataset
    exclusion_list = []
    if args.exclude_errors:
        exclusion_list = ["ERROR", "Error", "error", "Status", "Warning", "auditd", "audispd"]

    data_file_list = gather_files("data")
    # extract_archives(data_file_list, "data/")
    # extract_archives(label_file_list, "labels/")
    extract_training_set(data_file_list, exclusion_list)
    extract_testing_set(data_file_list, exclusion_list)
    inject_testing_set(data_file_list, 5)

    # Delete unzipped files
    for file in data_file_list:
        with open(os.path.join("data", file), encoding="utf-8"):
            if file.endswith("audit.log"):
                remove_unzipped_file(os.path.join("data", file))
                remove_unzipped_file(os.path.join("labels", file))

if __name__ == "__main__":
    main()
