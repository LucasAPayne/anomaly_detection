"""
Script to take in .ttl files produced by SLOGERT and assign labels to them, corresponding to how
suspicious an activity is.
Note: 4 indicates observed during training (not suspicious), and 0 indicates very suspicious
"""

import argparse
from plistlib import InvalidFileException


def load_content(path: str) -> list[str]:
    lines = []
    line_set = []
    with open(path, "r", encoding="utf-8") as infile:
        for line in infile:
            line_set.append(line)
            if line == '\n' and len(line_set) != 0:
                lines.append(line_set)
                line_set = []

    return lines


def main():
    # Define CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", "-i", help="the name of the input .ttl file")
    args = parser.parse_args()

    if not args.infile.endswith("train.ttl") and not args.infile.endswith("test.ttl"):
        raise InvalidFileException("File must be train.ttl or test.ttl")

    lines = load_content(args.infile)
    for line_set in lines:
        label_value = -1
        # Line always gets observed label if it is in training data
        if args.infile.endswith("train.ttl"):
            label_value = 4
        else:
            for line in line_set:
                if line.startswith("@prefix"):
                    break
                if "\\t\\t0" in line:
                    label_value = 0
                    break
                if "\\t\\t4" in line:
                    label_value = 4
                    break

            for index, line in enumerate(line_set):
                # line ending in > indicates it contains the subject, so skip
                # Also discard lines declaring prefixes and blank lines
                if line.startswith('<') or line.startswith("@prefix") or line == '\n':
                    continue
                else:
                    line_set[index] = line_set[index].rstrip() + '\t' + str(label_value) + '\n'

    with open(args.infile, "w", encoding="utf-8") as infile:
        for line_set in lines:
            infile.writelines(line_set)


if __name__ == "__main__":
    main()
