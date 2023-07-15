from dataclasses import dataclass
from functools import total_ordering
import os

@dataclass(eq=False)
@total_ordering
class ResultSet:
    hits_left_1:   float = 0.0
    hits_right_1:  float = 0.0
    hits_1:        float = 0.0
    hits_left_10:  float = 0.0
    hits_right_10: float = 0.0
    hits_10:       float = 0.0
    mr_left:       float = 0.0
    mr_right:      float = 0.0
    mr:            float = 0.0
    mrr_left:      float = 0.0
    mrr_right:     float = 0.0
    mrr:           float = 0.0

    def __eq__(self, other: object) -> bool:
        return self.mrr == other.mrr

    def __gt__(self, other: object) -> bool:
        return self.mrr > other.mrr

    def get_value(self, line: str) -> float:
        return float(line[line.rfind(":")+2:].rstrip())

    def get_result(self, line: str):
        if line.startswith("Hits left @1:"):
            self.hits_left_1 = self.get_value(line)
        elif line.startswith("Hits right @1:"):
            self.hits_right_1 = self.get_value(line)
        elif line.startswith("Hits @1:"):
            self.hits_1 = self.get_value(line)
        elif line.startswith("Hits left @10:"):
            self.hits_left_10 = self.get_value(line)
        elif line.startswith("Hits right @10:"):
            self.hits_right_10 = self.get_value(line)
        elif line.startswith("Hits @10:"):
            self.hits_10 = self.get_value(line)
        elif line.startswith("Mean rank left:"):
            self.mr_left = self.get_value(line)
        elif line.startswith("Mean rank right:"):
            self.mr_right = self.get_value(line)
        elif line.startswith("Mean rank:"):
            self.mr = self.get_value(line)
        elif line.startswith("Mean reciprocal rank left:"):
            self.mrr_left = self.get_value(line)
        elif line.startswith("Mean reciprocal rank right:"):
            self.mrr_right = self.get_value(line)
        elif line.startswith("Mean reciprocal rank:"):
            self.mrr = self.get_value(line)

def collect_results(results_dir: str) -> None:
    for root, _, files in os.walk(results_dir):
        for file in files:
            print(os.path.join(root,file))
            if not file.endswith(".log"):
                continue

            path = os.path.join(root, file)

            best_results = ResultSet()
            with open(path, "r+", encoding="utf-8") as result_file:
                # If file has already been processed, skip
                # broken = False
                # for line in result_file:
                #     if line.startswith("Best Results"):
                #         print(f"Best results already collected from {path}. Skipping\n")
                #         broken = True
                #         break
                # if broken:
                #     break

                result_file.seek(0)
                current_results = ResultSet()
                test_evaluation = False
                # Get set of results
                for line_num, line in enumerate(result_file):
                    # Only consider test results, disregard validation
                    if line.startswith("test_evaluation"):
                        test_evaluation = True

                        # If current MRR is best, this is the best set so far
                        if current_results > best_results:
                            best_results = current_results

                        current_results = ResultSet()
                    elif line.startswith("dev_evaluation"):
                        test_evaluation = False
                    if not test_evaluation:
                        continue

                    print(str(line_num) + "\t" + line)
                    current_results.get_result(line)

                result_file.write("\n--------------------------------------------------\n")
                result_file.write("Best Results\n")
                result_file.write("--------------------------------------------------\n\n")
                result_file.write(f"Hits left @1: {best_results.hits_left_1}\n")
                result_file.write(f"Hits right @1: {best_results.hits_right_1}\n")
                result_file.write(f"Hits @1: {best_results.hits_1}\n")
                result_file.write(f"Hits left @10: {best_results.hits_left_10}\n")
                result_file.write(f"Hits right @10: {best_results.hits_right_10}\n")
                result_file.write(f"Hits @10: {best_results.hits_10}\n")
                result_file.write(f"Mean rank left: {best_results.mr_left}\n")
                result_file.write(f"Mean rank right: {best_results.mr_right}\n")
                result_file.write(f"Mean rank: {best_results.mr}\n")
                result_file.write(f"Mean reciprocal rank left: {best_results.mrr_left}\n")
                result_file.write(f"Mean reciprocal rank right: {best_results.mrr_right}\n")
                result_file.write(f"Mean reciprocal rank: {best_results.mrr}\n")

def find_best_model(results_dir: str) -> None:
    results = {}
    for root, _, files in os.walk(results_dir):
        for file in files:
            if not file.endswith(".log"):
                continue

            path = os.path.join(root, file)
            with open(path, "r", encoding="utf-8") as results_file:
                result = ResultSet()
                # Seek to best set
                best_set = False
                for line in results_file:
                    if line.startswith("Best Results"):
                        best_set = True
                    if not best_set:
                        continue

                    result.get_result(line)

                results[path] = result

    best_model = max(results, key=results.get)
    best_mrr = results[best_model].mrr
    print(f"The best model is {best_model}, with an MRR of {best_mrr}\n")
