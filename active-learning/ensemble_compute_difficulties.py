import os
import pandas as pd

def get_iterations(directory):
    """Extract unique iteration numbers from file names in the directory."""
    iterations = set()
    for filename in os.listdir(directory):
        if filename.startswith("ensemble_results_"):
            # Extract iteration number from the filename
            try:
                iteration = int(filename.split('_')[2].split('.')[0])
                iterations.add(iteration)
            except ValueError:
                # Skip files that don't have a number after "ensemble_results_"
                pass
    return sorted(iterations)

def read_files(directory, iteration, language):
    """Read the ensemble_results and difficulty files for a given iteration."""
    ensemble_file = os.path.join(directory, f"ensemble_results_{iteration}.tsv")
    difficulty_file = os.path.join(directory, f"{language}_difficulty_{iteration}.tsv")
    ensemble_df = pd.read_csv(ensemble_file, sep='\t')
    difficulty_df = pd.read_csv(difficulty_file, sep='\t')
    return ensemble_df, difficulty_df

def calculate_difficulty_counts(difficulty_df):
    difficulty_names = {1: "Both", 2: "lemma", 3: "feature", 4: "neither"}
    counts = {}
    for difficulty, name in difficulty_names.items():
        counts[name] = len(difficulty_df[difficulty_df['level of difficulty'] == difficulty])
    return counts

def save_counts_to_tsv(counts, save_path):
    """Save the difficulty counts to a TSV file."""
    with open(save_path, 'w') as file:
        file.write("test result name\tboth\tlemma\tfeats\tneither\n")
        for iteration, values in counts.items():
            line = f"test_{iteration}\t{values['Both']}\t{values['lemma']}\t{values['feature']}\t{values['neither']}\n"
            file.write(line)

def process_difficulties(directory, language):
    difficulty_counts = {}
    iterations = get_iterations(directory)

    for iteration in iterations:
        _, difficulty_df = read_files(directory, iteration, language)
        # Difficulty counts calculation
        counts = calculate_difficulty_counts(difficulty_df)
        difficulty_counts[iteration] = counts

    return difficulty_counts

def main(language, criteria):
    directory_path = f'../experiments/ensemble_{criteria}/{language}'
    difficulty_counts = process_difficulties(directory_path, language)

    # Save difficulty counts
    difficulty_counts_save_path = os.path.join(f'../experiments/ensemble_{criteria}', f'{language}_difficulty_counts_ensemble_{criteria}.tsv')
    save_counts_to_tsv(difficulty_counts, difficulty_counts_save_path)
    print(f"Difficulty counts saved to {difficulty_counts_save_path}")


if __name__ == '__main__':
    # Take languages as user input
    languages = input("Please enter the language codes separated by commas (e.g. khk,kor,eng): ").strip().split(',')

    # Take criteria as user input
    criteria_list = input("Please enter the criteria separated by commas (e.g. entropy,edit_distance): ").strip().split(',')

    if not languages or '' in languages:
        print("Language codes cannot be empty.")
        exit(1)

    if not criteria_list or '' in criteria_list:
        print("Criteria cannot be empty.")
        exit(1)

    for language in languages:
        for criteria in criteria_list:
            print(f"Processing for language: {language} and criteria: {criteria}")
            main(language.strip(), criteria.strip())
