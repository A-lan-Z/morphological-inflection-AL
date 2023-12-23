import os
import csv
import sys

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

def compute_difficulties(difficulty_file, correct_indices, dist_values):
    difficulties_total = {i: 0 for i in range(1, 5)}

    with open(difficulty_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)  # skip header
        for idx, row in enumerate(reader, 1):
            difficulty = int(row[1])
            difficulties_total[difficulty] += 1

    return difficulties_total

def process_directory_for_difficulties(directory, difficulty_output_file):
    difficulty_results = []

    filenames = sorted(os.listdir(directory), key=lambda x: int(x.split('_')[-1].split('.tsv')[0]) if x.endswith(".tsv") and '.decode.test_' in x else 0)

    for filename in filenames:
        if filename.endswith(".tsv") and '.decode.test_' in filename:
            test_name = filename.split('.decode.')[1].split('.tsv')[0]
            test_num = test_name.split('_')[-1]
            difficulty_file_path = f"../experiments/experiment_{experiment}/{language}/{seed}/{seed}_{language}_difficulty_{test_num}.tsv"
            correct_indices, dist_values = get_correct_indices_and_dists(os.path.join(directory, filename))
            difficulties_total = compute_difficulties(difficulty_file_path, correct_indices, dist_values)
            difficulty_results.append(['total_instances_' + test_name.split('_')[-1]] + list(difficulties_total.values()))

    with open(difficulty_output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['test result name', 'both', 'lemma', 'feats', 'neither'])
        writer.writerows(difficulty_results)

    print(f"Difficulty results written to {difficulty_output_file}")

def get_filename_suffix(experiment):
    if experiment == "1":
        return "_random.tsv"
    elif experiment == "3":
        return "_entropy.tsv"
    elif experiment == "ID":
        return "_info_density.tsv"
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
        for seed in seeds:
            for experiment in experiments:
                directory_path = f"../experiments/experiment_{experiment.strip()}/{language.strip()}/{seed.strip()}"
                suffix = get_filename_suffix(experiment.strip())
                difficulty_output_file_path = f"../experiments/experiment_{experiment.strip()}/{language.strip()}/{seed.strip()}_{language.strip()}_difficulties{suffix}"
                process_directory_for_difficulties(directory_path, difficulty_output_file_path)
