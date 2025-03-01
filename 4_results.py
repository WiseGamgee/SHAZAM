import os
import re
import csv
import yaml
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from shazam.metrics import (get_metrics, get_monthly_thresholds, apply_thresholds,
                            get_binary_metrics, get_circular_polynomial_thresholds,
                            plot_anomaly_scores, plot_residuals)


def main():
    # Import test configuration file from local directory
    with open("0_config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)

    # Get model parameters from configuration file
    model_params = config["MODEL_PARAMS"]

    # Get results folder name using key params
    pfx = config["RESULTS_DIR_PREFIX"]
    tp = config["TIME_PERIOD"]
    lr = model_params["learning_rate"]
    lf = model_params["loss_function"]
    norm = config["NORMALISE"]

    # Create local results directory if folder is missing
    results_dir = f"results/{pfx}_{tp}_{lr}_{lf}/metrics"
    if norm:
        results_dir = f"results/{pfx}_{tp}_{lr}_{lf}_norm/metrics"
    if os.path.exists(results_dir) is False:
        os.makedirs(results_dir)

    # Save hyperparams as file
    config_dir = f"{results_dir}/configuration_file.yml"
    with open(config_dir, 'w') as hp_file:
        yaml.dump(config, hp_file, default_flow_style=False)

    # Load sdim dictionaries
    with open(f"results/{pfx}_{tp}_{lr}_{lf}_norm/validate/sdim_validation_dict", 'rb') as file:
        train_sdim_dict = pickle.load(file)
    with open(f"results/{pfx}_{tp}_{lr}_{lf}_norm/test/sdim_test_dict", 'rb') as file:
        test_sdim_dict = pickle.load(file)

    # Calculate threshold
    train_vals = list(train_sdim_dict.values())
    threshold = np.mean(train_vals) + 1.64 * np.std(train_vals)

    # Load labels as dictionary from csv
    with open(config["LABEL_DIR"], mode='r') as file:
        csv_reader = csv.reader(file)
        anomaly_label_dict = {}
        for row in csv_reader:
            name, label = row[0], float(row[1])
            date = re.search(r'\d{4}-\d{2}-\d{2}', name)[0]
            anomaly_label_dict[date] = label

    # Get matching lists of predictions and labels
    common_keys = test_sdim_dict.keys() & anomaly_label_dict.keys()
    y_pred = [test_sdim_dict[key] for key in common_keys]
    y_true = [anomaly_label_dict[key] for key in common_keys]

    # Get metrics using constant threshold
    metrics_dict = get_metrics(y_pred=y_pred,
                               y_true=y_true,
                               threshold=threshold)
    pre_deseason_metrics_dict = metrics_dict["threshold"]
    pre_deseason_metrics_dict["auc"] = metrics_dict["roc"]["auc"]
    pre_deseason_metrics_dict["aupr"] = metrics_dict["pr_curve"]["aupr"]
    pre_deseason_metrics_dict["random_aupr"] = metrics_dict["pr_curve"]["random_aupr"]

    # Get daily thresholds using polynomial
    poly_thresholds = get_circular_polynomial_thresholds(train_sdim_dict,
                                                         degree=1,
                                                         stddev_degree=1,
                                                         n_stddev=1.64,
                                                         results_dir=results_dir)

    y_pred_deseasoned_dict = apply_thresholds(scores_dict=test_sdim_dict,
                                              threshold_dict=poly_thresholds,
                                              threshold_time_period="day",
                                              return_binary=False)

    # Get metrics using constant threshold
    metrics_dict = get_metrics(y_pred=[y_pred_deseasoned_dict[key] for key in common_keys],
                               y_true=y_true,
                               threshold=1e-9)
    post_deseason_metrics_dict = metrics_dict["threshold"]
    post_deseason_metrics_dict["auc"] = metrics_dict["roc"]["auc"]
    post_deseason_metrics_dict["aupr"] = metrics_dict["pr_curve"]["aupr"]
    post_deseason_metrics_dict["random_aupr"] = metrics_dict["pr_curve"]["random_aupr"]

    binary_test_dict = apply_thresholds(scores_dict=test_sdim_dict,
                                        threshold_dict=poly_thresholds,
                                        threshold_time_period="day")
    y_pred_binary = [binary_test_dict[key] for key in common_keys]
    poly_threshold_metrics = get_binary_metrics(y_pred_binary=y_pred_binary,
                                                y_true=y_true)

    # Save dataframe of all metrics
    metrics_df = pd.DataFrame({"constant": pre_deseason_metrics_dict,
                               "polynomial": post_deseason_metrics_dict})
    metrics_df.to_csv(f'{results_dir}/threshold_metrics.csv', index=True)

    # Plot ROC Curve
    data = pd.DataFrame({"fpr": metrics_dict["roc"]["fpr"],
                         "tpr": metrics_dict["roc"]["tpr"]})

    # Set the Seaborn style
    sns.set(style="whitegrid", palette="pastel")
    plt.figure(figsize=(10, 10))

    # Plot ROC with AUC in legend
    auc = str(np.round(metrics_dict["roc"]["auc"], 4))
    sns.lineplot(data=data,
                 x="fpr",
                 y="tpr",
                 linewidth=2.5,
                 color='b',
                 errorbar=None,
                 label=f"AUC: {auc}")

    # Plot baseline model (AUC: 0.5)
    plt.plot([0, 1],
             [0, 1],
             linestyle='--',
             linewidth=2,
             label='Random Guess: 0.5',
             color='r')
    plt.xlabel("False Positive Rate", fontsize=14, fontweight='bold')
    plt.ylabel("True Positive Rate", fontsize=14, fontweight='bold')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, loc='best')
    plt.savefig(f"{results_dir}/roc_curve.png", format="png", bbox_inches='tight')

    # Plot PR curve
    data = pd.DataFrame({"Recall": metrics_dict["pr_curve"]["recall"],
                         "Precision": metrics_dict["pr_curve"]["precision"]})

    # Set the Seaborn style
    sns.set(style="whitegrid", palette="pastel")
    plt.figure(figsize=(10, 10))

    # Plot PR with average precision in label
    ap = str(np.round(metrics_dict["pr_curve"]["aupr"], 4))
    sns.lineplot(data=data,
                 x="Recall",
                 y="Precision",
                 linewidth=2.5,
                 color='b',
                 errorbar=None,
                 label=f"Avg. Prec: {ap}")

    # Plot baseline precision (precision frequency)
    baseline_precision = str(np.round(metrics_dict["pr_curve"]["random_aupr"], 4))
    plt.axhline(y=baseline_precision,
                color='r',
                linestyle='--',
                linewidth=2,
                label=f'Random Guess: {str(baseline_precision)}')
    plt.xlabel("Recall", fontsize=14, fontweight='bold')
    plt.ylabel("Precision", fontsize=14, fontweight='bold')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, loc='best')
    plt.savefig(f"{results_dir}/pr_curve.png", format="png", bbox_inches='tight')

    # Plot anomaly scores across entire dataset
    plot_anomaly_scores(train_sdim_dict,
                        test_sdim_dict,
                        anomaly_label_dict,
                        poly_thresholds,
                        results_dir)

    # Save residual plot
    plot_residuals(train_sdim_dict,
                   test_sdim_dict,
                   anomaly_label_dict,
                   poly_thresholds,
                   results_dir)


if __name__ == '__main__':
    main()
