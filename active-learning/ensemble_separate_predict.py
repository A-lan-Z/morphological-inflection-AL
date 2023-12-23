import os
import torch.nn.functional as F
import torch
import csv
import re
import sys


def read_single_tsv_file(directory, filename):
    file_data = []
    with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            # Skip the header row
            if row[0] == "target":
                print(f"Reading file: {filename}. Header: {row}")
                continue
            file_data.append(row)
    return file_data


def log_probs_to_probs(log_probs):
    return F.softmax(torch.tensor(log_probs), dim=0).tolist()


def calculate_entropy(probs):
    return -sum(p * torch.log(p) for p in probs if p >= 0.05)


def process_file_data(file_data, use_log_probs=False):
    results = []
    correct_predictions = 0

    for row in file_data:
        predictions = row[1].split('|')
        denorm_probs = [float(p) for p in row[3].split('|')]
        if use_log_probs:
            log_probs = row[2].split('|')
            probs = log_probs_to_probs([float(lp) for lp in log_probs])
        else:
            probs = denorm_probs

        best_prediction_index = probs.index(max(probs))
        best_prediction = predictions[best_prediction_index]
        best_prob = probs[best_prediction_index]
        target = row[0]
        edit_dist = int(row[4].split('|')[best_prediction_index]) - 2  # remove edit distance for BOS_IDX & EOS_IDX

        if best_prediction == target:
            correct_predictions += 1

        entropy = float(calculate_entropy(torch.tensor(denorm_probs))) # Convert tensor to float

        # loss and nll are filled with "NA"
        results.append((best_prediction, target, "NA", edit_dist, "NA", entropy))

    accuracy = correct_predictions / len(file_data)
    return results, accuracy


def main(directory, i, use_log_probs=False):
    pattern = re.compile(rf".*test_{i}_\d+\.tsv$")
    for filename in os.listdir(directory):
        if pattern.search(filename):
            file_data = read_single_tsv_file(directory, filename)
            results, accuracy = process_file_data(file_data, use_log_probs)

            seed = filename.split('_')[-1].split('.')[0]  # Extract the seed from the filename
            output_directory = os.path.join(directory, seed)
            os.makedirs(output_directory, exist_ok=True)

            base_name_without_extension = filename.rsplit('.', 1)[0]  # This will give "kor_al.decode.test_1_506"
            base_name_without_seed = '_'.join(base_name_without_extension.split('_')[:-1])  # Remove the seed
            output_filename = f"{seed}_{base_name_without_seed}.tsv"  # Construct the correct filename

            with open(os.path.join(output_directory, output_filename), 'w', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(['prediction', 'target', 'loss', 'dist', 'nll', 'entropy']) # Update CSV columns
                for result in results:
                    writer.writerow(result)

            print(f"Processed {filename}. Accuracy: {accuracy * 100:.2f}%")



if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script_name.py <learning_step> <use_log_probs>")
        sys.exit(1)

    i = sys.argv[1]
    use_log_probs = sys.argv[2].lower() == 'true'
    directory = "experiments/ensemble_entropy/khk"
    main(directory, i, use_log_probs)
