import numpy as np
import csv
import os
import re
import sys

def custom_tokenizer(s):
    """Tokenize the string, treating sequences wrapped in <> as single tokens."""
    return re.findall(r'<[^>]+>|.', s)

def edit_distance(str1, str2):
    """Simple Levenshtein implementation for evalm."""
    str1_tokens = custom_tokenizer(str1)
    str2_tokens = custom_tokenizer(str2)
    table = np.zeros([len(str2_tokens) + 1, len(str1_tokens) + 1])
    for i in range(1, len(str2_tokens) + 1):
        table[i][0] = table[i - 1][0] + 1
    for j in range(1, len(str1_tokens) + 1):
        table[0][j] = table[0][j - 1] + 1
    for i in range(1, len(str2_tokens) + 1):
        for j in range(1, len(str1_tokens) + 1):
            if str1_tokens[j - 1] == str2_tokens[i - 1]:
                dg = 0
            else:
                dg = 1
            table[i][j] = min(
                table[i - 1][j] + 1, table[i][j - 1] + 1, table[i - 1][j - 1] + dg
            )
    return int(table[len(str2_tokens)][len(str1_tokens)])

def correct_dist_values(filename):
    updated_rows = []
    print(f"Processing file: {filename}")
    with open(filename, 'r', newline='', encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter='\t')
        for row in reader:
            prediction = row['prediction'].replace(" ", "")
            target = row['target'].replace(" ", "")
            correct_dist = edit_distance(prediction, target)
            if correct_dist != int(row['dist']):
                print(f"Mismatch detected:")
                print(f"Prediction: {prediction}")
                print(f"Target: {target}")
                print(f"Old edit distance: {row['dist']}")
                print(f"New edit distance: {correct_dist}")
                print("------")
            row['dist'] = correct_dist
            updated_rows.append(row)

    # Write the updated rows back to the file
    with open(filename, 'w', newline='', encoding="utf-8") as file:
        fieldnames = ['prediction', 'target', 'average probability', 'entropy', 'dist']
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        writer.writerows(updated_rows)
    print(f"Finished processing file: {filename}")
    print("==================================")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script_name.py /path/to/directory/")
        sys.exit(1)

    directory = sys.argv[1]

    print("Starting the correction process...")
    # Correct the files
    for i in range(1, 26):
        filename = os.path.join(directory, f"ensemble_results_{i}.tsv")
        if os.path.exists(filename):
            correct_dist_values(filename)
    print("Finished the entire correction process!")
