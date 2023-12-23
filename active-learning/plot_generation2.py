import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys


def read_datasets(base_path, language_code, metric):
    data_file_names = {
        "ensemble_entropy": f"{language_code}_{metric}_ensemble_entropy.tsv",
        "ensemble_baseline": f"{language_code}_{metric}_ensemble_baseline.tsv",
        "single_model_entropy": f"{language_code}_average_{metric}_entropy.tsv"
    }

    # Check if files exist and add them to data_file_paths
    data_file_paths = {}
    for model, name in data_file_names.items():
        path = os.path.join(base_path, name)
        if os.path.exists(path):
            data_file_paths[model] = path

    # Read only existing files into dataframes
    dataframes = {name: pd.read_csv(path, sep='\t') for name, path in data_file_paths.items()}

    return dataframes


def plot_and_save(dataframes, output_folder, language_code, metric):
    if not dataframes:
        print(f"No data available for {language_code} in {metric}. Skipping plot generation.")
        return

    # Define plot metrics based on the metric type
    if metric == "difficulties":
        plot_metrics = ["both", "lemma", "feats", "neither"]
    else:
        plot_metrics = ["overall", "both", "lemma", "feats", "neither"]

    # Map for converting metric to its label for plot
    metric_labels = {
        "accuracies": "Accuracy",
        "dist": "Edit Distance",
        "difficulties": "Difficulty Category Count"
    }
    metric_label = metric_labels.get(metric, metric.replace('_', ' ').capitalize())

    num_data_points = len(next(iter(dataframes.values())))
    x_values = np.linspace(1000, 7000, num_data_points)

    for plot_metric in plot_metrics:
        accuracies = {}
        for model, df in dataframes.items():
            if metric == "dist" and model == "ensemble_baseline":
                # Add 2 to all data points for the ensemble_baseline model when the metric is 'dist'
                accuracies[model] = df[plot_metric]
            else:
                accuracies[model] = df[plot_metric]

        fig, ax = plt.subplots(figsize=(6, 4))

        for model, accuracy in accuracies.items():
            if model == "ensemble_entropy":
                ax.plot(x_values, accuracy, '-o', color="#d45087", label="EN-Test EN-Pool ", zorder=4)
            elif model == "ensemble_baseline":
                ax.plot(x_values, accuracy, color="#005aff", label="EN-Test SG-Pool", zorder=3)
            else:
                fmt = '--d' if model == "info_density" else '-^' if model == "random_baseline" else '-s'
                color = "#ffa600" if model == "info_density" else "#003f5c" if model == "random_baseline" else "#665191"
                label = "Information Density" if model == "info_density" else "Random Baseline" if model == "random_baseline" else "Single-Model Entropy Baseline"
                ax.plot(x_values, accuracy, fmt, color=color, label=label, zorder=2)

        if "random_baseline" in accuracies:
            full_training_set_baseline = accuracies["ensemble_entropy"].iloc[-1]
            ax.axhline(full_training_set_baseline, color='k', linestyle='dashed', alpha=0.5, label="Full training set Baseline", zorder=1)

        ax.set_xlabel("Training Instances")
        ax.set_ylabel(f"{metric_label} ({plot_metric.capitalize()})")
        ax.set_title(f"{language_code.upper()} - {plot_metric.capitalize()} {metric_label} Across Different Models")
        ax.legend()
        ax.grid(True, linestyle='--', which='both', color='gray', alpha=0.5)
        plt.tight_layout()

        output_file = os.path.join(output_folder, f"{language_code}_{metric}_{plot_metric}_plot.png")
        plt.savefig(output_file, format='png', dpi=300)
        plt.close()



def main():
    base_path = '../plot_generation'
    languages = input("Please enter the language codes separated by commas (e.g. ara,khk,kor,pol): ").strip().split(',')
    metrics = input("Please enter the metrics separated by commas (e.g. accuracies,dist,difficulties): ").strip().split(',')

    if not languages or not metrics:
        print("Language codes and metrics cannot be empty.")
        sys.exit(1)

    for language in languages:
        for metric in metrics:
            language_path = os.path.join(base_path, "entropy-ensemble-ensemble_baseline", language.strip(), metric.strip())
            dataframes = read_datasets(language_path, language.strip(),
                                       metric.strip())  # variance_dataframes is no longer returned
            plot_and_save(dataframes, language_path, language.strip(),
                          metric.strip())  # Removed variance_dataframes as argument


if __name__ == "__main__":
    main()
