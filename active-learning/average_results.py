import os
import csv
import sys


def calculate_variance(summed_data, square_sum_data, num_files):
    variance_data = {}
    for key in summed_data:
        averages = summed_data[key]
        squares = square_sum_data[key]
        variance_data[key] = [(sqr / num_files - (avg / num_files) ** 2) for sqr, avg in zip(squares, averages)]
    return variance_data


def average_files(files):
    summed_data = {}
    square_sum_data = {}  # To store the sum of squares
    num_files = len(files)

    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            header = next(reader)
            for row in reader:
                test_name = row[0]
                values = [float(val) for val in row[1:]]
                if test_name not in summed_data:
                    summed_data[test_name] = values
                    square_sum_data[test_name] = [val ** 2 for val in values]
                else:
                    summed_data[test_name] = [sum(x) for x in zip(summed_data[test_name], values)]
                    square_sum_data[test_name] = [sum(x) for x in
                                                  zip(square_sum_data[test_name], [val ** 2 for val in values])]

    averaged_data = {key: [x / num_files for x in summed_data[key]] for key in summed_data}
    variance_data = calculate_variance(summed_data, square_sum_data, num_files)

    return header, averaged_data, variance_data


def write_data_to_file(output_file, header, data):
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(header)
        for key, values in data.items():
            writer.writerow([key] + values)


def get_filename_suffix(experiment):
    if experiment == "1":
        return "_random.tsv"
    elif experiment == "3":
        return "_entropy.tsv"
    elif experiment == "ID":
        return "_info_density.tsv"
    else:
        return ".tsv"


def main(include_difficulty=True):
    languages = input("Please enter the language codes separated by commas (e.g. khk,kor_1st,eng): ").strip().split(',')
    experiments = input("Please enter the experiment numbers separated by commas: ").strip().split(',')

    for language in languages:
        for experiment in experiments:
            directory_path = f"../experiments/experiment_{experiment.strip()}/{language.strip()}"
            output_path = f"../experiments/experiment_{experiment.strip()}"

            # Get the filename suffix based on experiment
            suffix = get_filename_suffix(experiment.strip())

            # Collect all files for each type with the appropriate suffix
            accuracy_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if
                              f.endswith(f'_accuracies{suffix}')]

            if include_difficulty:
                difficulty_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if
                                    f.endswith(f'_difficulties{suffix}')]

            dist_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if
                          f.endswith(f'_dist{suffix}')]

            # Average the files
            accuracy_header, averaged_accuracy_data, variance_accuracy_data = average_files(accuracy_files)
            if include_difficulty:
                difficulty_header, averaged_difficulty_data, variance_difficulty_data = average_files(difficulty_files)
            dist_header, averaged_dist_data, variance_dist_data = average_files(dist_files)

            # Write the averaged data and variance data to new files
            write_data_to_file(os.path.join(output_path, f'{language}_average_accuracies{suffix}'), accuracy_header,
                               averaged_accuracy_data)
            write_data_to_file(os.path.join(output_path, f'{language}_average_accuracies_variance{suffix}'),
                               accuracy_header, variance_accuracy_data)

            if include_difficulty:
                write_data_to_file(os.path.join(output_path, f'{language}_average_difficulties{suffix}'),
                                   difficulty_header, averaged_difficulty_data)
                write_data_to_file(os.path.join(output_path, f'{language}_average_difficulties_variance{suffix}'),
                                   difficulty_header, variance_difficulty_data)

            write_data_to_file(os.path.join(output_path, f'{language}_average_dist{suffix}'), dist_header,
                               averaged_dist_data)
            write_data_to_file(os.path.join(output_path, f'{language}_average_dist_variance{suffix}'), dist_header,
                               variance_dist_data)

        print(f"Averaged files for {language} in experiment {experiment} have been written.")


if __name__ == "__main__":
    process_difficulty = input("Do you want to process difficulty files? (yes/no): ").lower() == "yes"
    main(process_difficulty)

