import calendar
import os
import shutil
from datetime import datetime


"""
Get timestamp from log line and return date
log_type: type of log file. Options are {access, error, audit, exim, suricata, auth, daemon, mail, mail-info, messages, sys, user}
line: line of audit log to be processed
"""
def get_date(log_type: str, line: str, ):
    if log_type == "access":
        timestamp = line[line.find("[")+1:line.find("[")+12]
        day = int(timestamp.split("/")[0])
        month = list(calendar.month_abbr).index(timestamp.split("/")[1])
        year = int(timestamp.split("/")[2])
        date = datetime(year, month, day)
        return date.strftime("%m/%d/%Y")

    elif log_type == "error":
        # Timestamp format: [Sun timestamp]
        timestamp = line[line.find("[")+5:line.find("]")]
        month = list(calendar.month_abbr).index(timestamp.split()[0])
        day = int(timestamp.split()[1])
        year = int(timestamp.split()[3])
        date = datetime(year, month, day)
        return date.strftime("%m/%d/%Y")

    elif log_type == "audit":
        # Timestamp format: msg=audit(timestamp:)
        timestamp = float(line[line.find("audit(")+6:line.find(":")])
        date = datetime.fromtimestamp(timestamp)
        return date.strftime("%m/%d/%Y")

    elif log_type == "mainlog":
        timestamp = line[:line.find(" ")]
        date = datetime.fromisoformat(timestamp)
        return date.strftime("%m/%d/%Y")

    elif log_type == "fast":
        return line[:line.find("-")]

    elif log_type == "auth" or log_type == "daemon" or log_type == "mail" or log_type == "messages" or log_type == "syslog" or log_type == "user":
        month = list(calendar.month_abbr).index(line.split()[0])
        day = int(line.split()[1])
        year = 2020
        date = datetime(year, month, day)
        return date.strftime("%m/%d/%Y")

"""
Return the path to the parent directory of a file
path: path to file
"""
def parent_dir(path: str):
    return path[:path.rfind("/")]

"""
Gather relevant files recursively from root_dir
Returns a list of file names
"""
def gather_files(root_dir: str):
    file_list = []
    for root, subdirs, files in os.walk(root_dir):
        for file in files:
            if not file.startswith(('log', 'attack')) and not file.endswith('json'):
                # Do not include root dir in file path to make it easier to use output paths
                file_list.append(os.path.join(root, file)[len(root_dir):])

    return file_list

"""
Wrapper around shutil.unpack_archive to include print statements
zip_path: path to zip archive
unzipped_name: name of extracted file
"""
def unzip(zip_path: str, unzipped_name: str):
    if not os.path.exists(unzipped_name) and not zip_path.endswith("log"):
        print("Unpacking {0}...".format(zip_path), end=' ', flush=True)
        # Extract to same path as zip file (e.g., mail.cup.com/audit)
        shutil.unpack_archive(zip_path, parent_dir(zip_path))
        print("Done")

"""
Wrapper around os.remove to include print statements, intended to be used on unzipped files
unzipped_path: path to (unzipped) file
"""
def remove_unzipped_file(unzipped_path: str):
    print("Removing unzipped file {0}...".format(unzipped_path), end=' ', flush=True)
    os.remove(unzipped_path)
    print("Done")


"""
Extract zip archives from a file list
file_list: list of files, where the file names do not include the root directory
root_dir: the root directory of the file list
"""
def extract_archives(file_list: list, root_dir: str):
    for file in file_list:
        # unzip function handles the case where the zip file was already extracted
        if file.endswith("zip") or file.endswith("audit.log"):
            zip_path = os.path.join(root_dir, file)
            unzipped_name = zip_path.replace("zip", "log")
            unzip(zip_path, unzipped_name)
    

"""
Extract training over a 1-day period
Attacks occur on 03/04/2020 and 03/05/2020
data_file_list: list of all file names to be processed, without the root directory (e.g., data/)
"""
def extract_training_set(root_dir: str, data_file_list: list):
    max_size = 5 * 1024
    for file in data_file_list:
        if file.endswith("zip"):
            continue

        data_file_to_open = os.path.join(root_dir, "data/", file)
        out_file_to_open = os.path.join("output/train/", file)
        # If there is not already a folder for output, create one (make sure not to make the target output file into a directory)
        if not os.path.exists(parent_dir(out_file_to_open)):
            os.makedirs(parent_dir(out_file_to_open))

        with open(data_file_to_open, "r") as in_file, open(out_file_to_open, "w") as out_file:
            print("Extracting training data from {0}...".format(file), end=' ', flush=True)

            # To find the log type, return the name found between the last slash and the last period
            # If the log file contains the server name (e.g., mail.cup.com-access), remove the last period and everything before
            # Also, remove the "com-" part
            log_type = file[file.rfind("/")+1:]
            log_type = log_type.replace(".log", "").replace(".info", "")
            log_type = log_type[log_type.rfind(".")+1:]
            log_type = log_type.replace("com-", "")
            for line in in_file:
                if get_date(log_type, line) == "03/03/2020" and out_file.tell() < max_size:
                    out_file.write(line)
            
            print("Done")
    
    print("Training data extracted")


"""
Extract all attack data by looping through each line of each file and comparing to the same label file
Attacks occur on 03/04/2020 and 03/05/2020
root_dir: root directory of dataset
data_file_list: list of all file names to be processed, without the root directory (e.g., data/)
"""
def extract_test_val_set(root_dir: str, data_file_list: list):
    max_size = 512
    for file in data_file_list:
        if file.endswith("zip"):
            continue

        label_file_to_open = os.path.join(root_dir, "labels/", file)
        data_file_to_open = os.path.join(root_dir, "data/", file)
        test_file_to_open = os.path.join("output/test/", file)
        val_file_to_open = os.path.join("output/valid/", file)

        # If there is not already a folder for output, create one (make sure not to make the target output file into a directory)
        if not os.path.exists(parent_dir(test_file_to_open)):
            os.makedirs(parent_dir(test_file_to_open))

        if not os.path.exists(parent_dir(val_file_to_open)):
            os.makedirs(parent_dir(val_file_to_open))

        with open(label_file_to_open, "r") as label_file, open(data_file_to_open, "r") as data_file, open(test_file_to_open, "w") as test_file, open(val_file_to_open, "w") as val_file:
            print("Extracting attack data from {0}...".format(file), end=' ', flush=True)

            # Write half of the attacks to the training set and the other half to the validation set
            for line_num, (label_line, data_line) in enumerate(zip(label_file, data_file)):
                if label_line.strip() != "0,0":
                    if line_num % 2 == 0 and test_file.tell() < max_size:
                        test_file.write(data_line.rstrip() + "\t\t0\n")
                    elif line_num % 2 != 0 and val_file.tell() < max_size:
                        val_file.write(data_line.rstrip() + "\t\t0\n")
                        
            print("Done")

    print("Attack data extracted")


"""
Put the last few lines of each file in the training set into the testing set
data_file_list: list of all file names to be processed, without the root directory (e.g., data/)
lines: the number of lines to copy from each training file to each testing file
"""
def inject_test_val_set(data_file_list: list, lines: int):
    print("Injecting training data into test and validation sets...", end=' ', flush=True)
    for file in data_file_list:
        if file.endswith("zip"):
            continue

        train_file_to_open = os.path.join("output/train/", file)
        test_file_to_open = os.path.join("output/test/", file)
        val_file_to_open = os.path.join("output/valid/", file)
        with open(train_file_to_open, "r+") as in_file, open(test_file_to_open, "a") as test_file, open(val_file_to_open, "a") as val_file:
            train_lines = in_file.readlines()

            # Append an "observed during training" label to each line from the training set
            # Split data between test and val sets
            lines_to_write = train_lines[-lines:]
            for line_num, line in enumerate(lines_to_write):
                if line_num % 2 == 0:
                    test_file.write(line.rstrip() + "\t\t4\n")
                else:
                    val_file.write(line.rstrip() + "\t\t4\n")
                                    
    print("Done")


def main():
    dataset_path = "AIT-LDS-v1_1"
    data_file_list = gather_files(os.path.join(dataset_path, "data/"))
    label_file_list = gather_files(os.path.join(dataset_path, "labels/"))
    # extract_archives(data_file_list, "data/")
    # extract_archives(label_file_list, "labels/")
    extract_training_set(dataset_path, data_file_list)
    extract_test_val_set(dataset_path, data_file_list)
    inject_test_val_set(data_file_list, 5)

    # Delete unzipped files
    # for file in data_file_list:
    #     with open(os.path.join("data/", file)):
    #         if file.endswith("audit.log"):
    #             remove_unzipped_file(os.path.join("data/", file))
    #             remove_unzipped_file(os.path.join("labels/", file))


if __name__ == "__main__":
    main()
