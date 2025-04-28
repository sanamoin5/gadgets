import numpy as np
import os
import csv


def compute_rmse(preds, targets):
    return np.sqrt(np.mean((np.array(preds) - np.array(targets)) ** 2))


def save_metrics(metrics_list, output_dir):
    """
    Save training, validation, and test metrics to a CSV file for visualization.

    Args:
        metrics_list (list of dict): Each dict contains metrics from an epoch or test stage.
        output_dir (str): Directory where the CSV file will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "metrics_sbert_nv.csv")
    if not metrics_list:
        return

    # Use the keys from the first dict as headers.
    headers = metrics_list[0].keys()
    with open(file_path, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        for metric in metrics_list:
            writer.writerow(metric)
    print(f"Metrics saved to {file_path}")
