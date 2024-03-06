import io
import os

def flush_buffer_to_file(buffer: list, file: io.TextIOWrapper, chunk_size: int=None) -> None:
    """
    Flush a buffer to a file. If chunk_size is passed, check if the buffer is at least that
    big, and if it is flush it. If chunk_size is not passed, flush the whole buffer.
    """
    if chunk_size:
        if len(buffer) >= chunk_size:
            file.writelines(buffer)
            buffer.clear()
    else:
        file.writelines(buffer)
        buffer.clear()

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

    train_path = os.path.join(data_dir, "train", "hdfs.log")
    test_path = os.path.join(data_dir, "test", "hdfs.log")

    with open(train_path, "r+", encoding="utf-8") as train_file, \
         open(test_path, "a", encoding="utf-8") as test_file:
        train_lines = train_file.readlines()
        test_file.writelines(train_lines[-test_size:])
        train_file.writelines(train_lines[:-test_size])

    print("Done")


def extract_dataset(data_dir: str, chunk_size: int=10000) -> None:
    """
    Extract the HDFS dataset and break up into train, test, and val sets.

    Parameters
    ----------
    - `data_dir`: directory containing raw log file
    - `chunk_size`: the max number of lines to be loaded from the log file
      into memory at one time
    """
    train_buffer = []
    test_buffer = []

    if not os.path.exists(os.path.join(data_dir, "train")):
        os.mkdir(os.path.join(data_dir, "train"))
    if not os.path.exists(os.path.join(data_dir, "test")):
        os.mkdir(os.path.join(data_dir, "test"))

    data_file_path = os.path.join(data_dir, "raw", "hdfs.log")
    label_file_path = os.path.join(data_dir, "raw", "mlabel.txt")
    train_file_path = os.path.join(data_dir, "train", "hdfs.log")
    test_file_path = os.path.join(data_dir, "test", "hdfs.log")

    train_size = 0
    test_size = 0

    with open(data_file_path, "r", encoding="utf-8") as data_file, \
         open(label_file_path, "r", encoding="utf-8") as label_file, \
         open(train_file_path, "w", encoding="utf-8") as train_file, \
         open(test_file_path, "w", encoding="utf-8") as test_file:
        print(f"Extracting data from {data_file_path}...", end=' ', flush=True)
        for line, label in zip(data_file, label_file):
            if label.strip() == "0 -1" and train_size < 10_000: # Normal label
                train_buffer.append(line.strip() + "\t\t0\n")
                train_size += 1
            else:
                if test_size < 10_000:
                    test_buffer.append(line.strip() + "\t\t1\n")
                    test_size += 1

            flush_buffer_to_file(train_buffer, train_file, chunk_size)
            flush_buffer_to_file(test_buffer, test_file, chunk_size)

        flush_buffer_to_file(train_buffer, train_file)
        flush_buffer_to_file(test_buffer, test_file)
        print("Done")

    inject_testing_set(data_dir, test_size)
