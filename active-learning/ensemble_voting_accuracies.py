import os
import csv
import sys
from collections import defaultdict


def get_correct_indices_and_dists(test_file):
    correct_indices = []
    dist_values = {}
    with open(test_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)  # skip header
        for idx, row in enumerate(reader, 1):
            dist = float(row[3])  # dist column
            dist_values[idx] = dist
            if dist == 0:
                correct_indices.append(idx)
    return correct_indices, dist_values

def get_ensemble_predictions(base_directory, seeds, language, i):
    predictions_weights = defaultdict(lambda: defaultdict(float))

    for seed in seeds:
        directory = os.path.join(base_directory, language, seed)
        file_path = os.path.join(directory, f"{seed}_{language}_al.decode.test_{i}.tsv")

        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)  # skip header
            for idx, row in enumerate(reader, 1):
                prediction = row[0]
                entropy = float(row[5])
                # Use the inverse of entropy as weight
                weight = 1 / (1 + entropy)
                predictions_weights[idx][prediction] += weight

    # Get predictions with the highest cumulative weight for each test instance
    ensemble_predictions = {}
    for idx, predictions in predictions_weights.items():
        ensemble_predictions[idx] = max(predictions, key=predictions.get)

    return ensemble_predictions


def compute_accuracies(difficulty_file, correct_indices, dist_values):
    difficulties_correct = {i: 0 for i in range(1, 5)}
    difficulties_total = {i: 0 for i in range(1, 5)}
    difficulties_dist_sum = {i: 0.0 for i in range(1, 5)}  # Sum of dist values for each difficulty

    with open(difficulty_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)  # skip header
        for idx, row in enumerate(reader, 1):
            difficulty = int(row[1])
            difficulties_total[difficulty] += 1
            difficulties_dist_sum[difficulty] += dist_values.get(idx, 0)  # Add dist value to the sum
            if idx in correct_indices:
                difficulties_correct[difficulty] += 1

    overall_accuracy = len(correct_indices) / sum(difficulties_total.values())
    difficulty_accuracies = {i: difficulties_correct[i] / difficulties_total[i] for i in range(1, 5)}

    overall_dist = sum(dist_values.values()) / len(dist_values)
    difficulty_dists = {i: difficulties_dist_sum[i] / difficulties_total[i] for i in range(1, 5)}  # Average dist for each difficulty

    return overall_accuracy, difficulty_accuracies, difficulties_total, overall_dist, difficulty_dists


def process_directory(base_directory, seeds, language, accuracy_output_file):
    accuracy_results = []

    # Ensure the file naming convention holds
    filenames = sorted([f for f in os.listdir(os.path.join(base_directory, language, seeds[0])) if
                        f.endswith(".tsv") and '.decode.test_' in f],
                       key=lambda x: int(x.split('_')[-1].split('.tsv')[0]))

    for filename in filenames:
        if filename.endswith(".tsv") and '.decode.test_' in filename:
            test_name = filename.split('.decode.')[1].split('.tsv')[0]
            test_num = test_name.split('_')[-1]

            ensemble_predictions = get_ensemble_predictions(base_directory, seeds, language, test_num)
            difficulty_file_path = os.path.join(base_directory, language, f"{language}_difficulty_25.tsv")

            with open(os.path.join(base_directory, language, seeds[0], filename), 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t')
                next(reader)  # skip header
                correct_indices = []
                dist_values = {}
                for idx, row in enumerate(reader, 1):
                    target = row[1]
                    dist = float(row[3])  # dist column
                    dist_values[idx] = dist
                    if ensemble_predictions.get(idx) == target:
                        correct_indices.append(idx)

            overall_accuracy, difficulty_accuracies, _, _, _ = compute_accuracies(difficulty_file_path, correct_indices,
                                                                                  dist_values)
            accuracy_results.append([test_name, overall_accuracy] + list(difficulty_accuracies.values()))

    # Write accuracy results to {seed}_{language}_accuracies.tsv
    with open(accuracy_output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['test result name', 'overall accuracy', 'both', 'lemma', 'feats', 'neither'])
        writer.writerows(accuracy_results)

    print(f"Ensemble accuracy results written to {accuracy_output_file}")


def get_filename_suffix(experiment):
    if experiment == "1":
        return "_random_ensemble_voting.tsv"
    elif experiment == "3":
        return "_entropy_ensemble_voting.tsv"
    elif experiment == "ID":
        return "_info_density_ensemble_voting.tsv"
    else:
        return ".tsv"


if __name__ == "__main__":
    languages = input("Please enter the language codes separated by commas (e.g. khk,kor_1st,eng): ").strip().split(',')
    seeds = input("Please enter the seeds separated by commas: ").strip().split(',')
    experiments = input("Please enter the experiment numbers separated by commas: ").strip().split(',')

    if not languages:
        print("Language codes cannot be empty.")
        sys.exit(1)

    for language in languages:
        for experiment in experiments:
            base_directory = f"../experiments/experiment_{experiment.strip()}"
            suffix = get_filename_suffix(experiment.strip())
            accuracy_output_file_path = os.path.join(base_directory, f"{language}_ensemble_accuracies{suffix}")

            process_directory(base_directory, seeds, language, accuracy_output_file_path)


