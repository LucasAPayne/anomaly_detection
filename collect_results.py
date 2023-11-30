"""
Tools for finding and reporting the best results from a KGC classification task.
"""

from dataclasses import dataclass
from functools import total_ordering
import os

@dataclass(eq=False)
@total_ordering
class ResultSet:
    """
    A dataclass to hold a full set of results for a KGC classification task
    """
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

    accuracy:  float = 0.0
    f1_score:  float = 0.0
    precision: float = 0.0
    recall:    float = 0.0
    support:   float = 0.0
    tp:        int = 0
    tn:        int = 0
    fp:        int = 0
    fn:        int = 0
    tpr:       float = 0.0
    fpr:       float = 0.0
    tnr:       float = 0.0
    fnr:       float = 0.0

    def __eq__(self, other: object) -> bool:
        return self.f1_score == other.f1_score

    def __gt__(self, other: object) -> bool:
        return self.f1_score > other.f1_score

    def get_value(self, line: str) -> float:
        """
        Get a metric value from a line.
        Lines look like "Metric: value".

        Parameters
        ----------
        - `line`: the line to parse 
        """
        content = line[line.rfind(":")+2:].rstrip()
        result = 0.0
        if "None" not in content:
            result = float(content)
        return result

    def get_result(self, line: str):
        """
        Fill one ResultSet by getting values for each type of result

        Parameters
        ----------
        - `line`: the line to parse
        """
        if line.startswith("Hits Left @1:"):
            self.hits_left_1 = self.get_value(line)
        elif line.startswith("Hits Right @1:"):
            self.hits_right_1 = self.get_value(line)
        elif line.startswith("Hits @1:"):
            self.hits_1 = self.get_value(line)
        elif line.startswith("Hits Left @10:"):
            self.hits_left_10 = self.get_value(line)
        elif line.startswith("Hits Right @10:"):
            self.hits_right_10 = self.get_value(line)
        elif line.startswith("Hits @10:"):
            self.hits_10 = self.get_value(line)
        elif line.startswith("Mean Rank left:"):
            self.mr_left = self.get_value(line)
        elif line.startswith("Mean Rank right:"):
            self.mr_right = self.get_value(line)
        elif line.startswith("Mean Rank:"):
            self.mr = self.get_value(line)
        elif line.startswith("Mean Reciprocal Rank Left:"):
            self.mrr_left = self.get_value(line)
        elif line.startswith("Mean Reciprocal Rank Right:"):
            self.mrr_right = self.get_value(line)
        elif line.startswith("Mean Reciprocal Rank:"):
            self.mrr = self.get_value(line)

        elif line.startswith("Accuracy:"):
            self.accuracy = self.get_value(line)
        elif line.startswith("F1-Score:"):
            self.f1_score = self.get_value(line)
        elif line.startswith("Precision:"):
            self.precision = self.get_value(line)
        elif line.startswith("Recall:"):
            self.recall = self.get_value(line)
        elif line.startswith("Support:"):
            self.support = self.get_value(line)
        elif line.startswith("True Positives:"):
            self.tp = self.get_value(line)
        elif line.startswith("False Positives:"):
            self.fp = self.get_value(line)
        elif line.startswith("True Negatives:"):
            self.tn = self.get_value(line)
        elif line.startswith("False Negatives:"):
            self.fn = self.get_value(line)
        elif line.startswith("True Positive Rate:"):
            self.tpr = self.get_value(line)
        elif line.startswith("True Negative Rate:"):
            self.tnr = self.get_value(line)
        elif line.startswith("False Negative Rate:"):
            self.fnr = self.get_value(line)

def collect_results(results_dir: str) -> None:
    """
    Collect the best ResultSet from each metric.log file in a directory
    obtained by running a KGC classification task.

    Parameters
    ----------
    - `results_dir`: root directory containing metric.log files
    """
    for root, _, files in os.walk(results_dir):
        for file in files:
            print(os.path.join(root,file))
            if not file.endswith(".log"):
                continue

            path = os.path.join(root, file)

            best_results = ResultSet()
            with open(path, "r+", encoding="utf-8") as result_file:
                # If file has already been processed, skip
                broken = False
                for line in result_file:
                    if line.startswith("Best Results"):
                        print(f"Best results already collected from {path}. Skipping\n")
                        broken = True
                        break
                if broken:
                    break

                result_file.seek(0)
                current_results = ResultSet()
                test_evaluation = False
                # Get set of results
                for line in result_file:
                    # Only consider test results, disregard validation
                    if line.startswith("test_evaluation"):
                        test_evaluation = True

                        # If current F1-socre is best, this is the best set so far
                        if current_results > best_results:
                            best_results = current_results

                        current_results = ResultSet()
                    elif line.startswith("dev_evaluation"):
                        test_evaluation = False
                    if not test_evaluation:
                        continue

                    current_results.get_result(line)

                result_file.write("\n--------------------------------------------------\n")
                result_file.write("Best Results\n")
                result_file.write("--------------------------------------------------\n\n")
                result_file.write(f"Hits Left @1: {best_results.hits_left_1}\n")
                result_file.write(f"Hits Right @1: {best_results.hits_right_1}\n")
                result_file.write(f"Hits @1: {best_results.hits_1}\n")
                result_file.write(f"Hits Left @10: {best_results.hits_left_10}\n")
                result_file.write(f"Hits Right @10: {best_results.hits_right_10}\n")
                result_file.write(f"Hits @10: {best_results.hits_10}\n")
                result_file.write(f"Mean Rank Left: {best_results.mr_left}\n")
                result_file.write(f"Mean Rank Right: {best_results.mr_right}\n")
                result_file.write(f"Mean Rank: {best_results.mr}\n")
                result_file.write(f"Mean Reciprocal Rank Left: {best_results.mrr_left}\n")
                result_file.write(f"Mean Reciprocal Rank Right: {best_results.mrr_right}\n")
                result_file.write(f"Mean Reciprocal Rank: {best_results.mrr}\n")

                result_file.write("\n")
                result_file.write(f"Accuracy: {best_results.accuracy}\n")
                result_file.write(f"F1-score: {best_results.f1_score}\n")
                result_file.write(f"Precision: {best_results.precision}\n")
                result_file.write(f"Recall: {best_results.recall}\n")
                result_file.write(f"Support: {best_results.support}\n")
                result_file.write(f"True Positives: {best_results.tp}\n")
                result_file.write(f"False Positives: {best_results.fp}\n")
                result_file.write(f"True Negatives: {best_results.tn}\n")
                result_file.write(f"False Negatives: {best_results.fn}\n")
                result_file.write(f"True Positive Rate: {best_results.tpr}\n")
                result_file.write(f"False Positive Rate: {best_results.fpr}\n")
                result_file.write(f"True Negative Rate: {best_results.tnr}\n")
                result_file.write(f"False Negative Rate: {best_results.fnr}\n")


def find_best_model(results_dir: str) -> None:
    """
    Given multiple models run on one dataset, find the model with the best ResultSet and print it.

    Parameters
    ----------
    - `results_dir`: root directory containing metric.log files
    """
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
    best_accuracy = results[best_model].accuracy
    best_f1 = results[best_model].f1_score
    print(f"The best model is {best_model}, with an Accuracy of {best_accuracy} and an F1-score of \
          {best_f1}\n")
