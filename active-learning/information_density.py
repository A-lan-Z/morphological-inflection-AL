import numpy as np

# 1. Data Representation
def to_vector(lemma, charset, features, feature_set):
    vector = [0] * (len(charset) + len(feature_set))
    for char in lemma:
        if char in charset:
            vector[charset.index(char)] = 1
    for feature in features:
        if feature in feature_set:
            vector[len(charset) + feature_set.index(feature)] = 1
    return vector

# 2. Compute Cosine Similarity
def cosine_similarity(A, B):
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

# 2.1 Compute Edit Distance (Levenshtein Distance)
def edit_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for index2, char2 in enumerate(s2):
        new_distances = [index2 + 1]
        for index1, char1 in enumerate(s1):
            if char1 == char2:
                new_distances.append(distances[index1])
            else:
                new_distances.append(1 + min((distances[index1], distances[index1 + 1], new_distances[-1])))
        distances = new_distances
    return distances[-1]


# 3. Calculate Density Term
def compute_density(lemmas, charset, feature_set):
    print("Computing vectors for lemmas...")
    vectors = [to_vector(lemma[0], charset, lemma[1], feature_set) for lemma in lemmas]
    densities = []
    avg_lemma_sims = []
    avg_feature_sims = []
    print("Calculating density for each lemma...")
    for i, (lemma, vector) in enumerate(zip(lemmas, vectors)):
        lemma_similarities = [1 / (1 + edit_distance(lemma[0], other_lemma[0])) if lemma[0] != other_lemma[0] else 0 for
                              other_lemma in lemmas]
        feature_similarities = [cosine_similarity(vector, other_vector) if vector != other_vector else 0 for
                                other_vector in vectors]

        avg_lemma_sim = sum(lemma_similarities) / (len(lemmas) - 1)  # -1 to exclude self
        avg_feature_sim = sum(feature_similarities) / (len(lemmas) - 1)

        # Average the lemma and feature similarities
        overall_average_similarity = (avg_lemma_sim + avg_feature_sim) / 2

        avg_lemma_sims.append(avg_lemma_sim)
        avg_feature_sims.append(avg_feature_sim)
        densities.append(overall_average_similarity)

        # Print to terminal
        print(
            f"Index: {i}, Lemma: {lemma[0]}, Avg Lemma Sim: {avg_lemma_sim:.4f}, Avg Feature Sim: {avg_feature_sim:.4f}, Info Density: {overall_average_similarity:.4f}")

    return avg_lemma_sims, avg_feature_sims, densities

# 4. Store in a File
def save_to_file(lemmas, avg_lemma_sims, avg_feature_sims, densities, filename):
    print(f"Saving results to {filename}...")
    with open(filename, 'w', encoding='utf-8') as file:
        # Write headers
        file.write("index\tlemma\tavg_lemma_similarity\tavg_feature_similarity\tinformation_density\n")
        for index, (lemma, avg_lemma_sim, avg_feature_sim, density) in enumerate(zip(lemmas, avg_lemma_sims, avg_feature_sims, densities)):
            file.write(f"{index}\t{lemma}\t{avg_lemma_sim}\t{avg_feature_sim}\t{density}\n")
    print("Results saved successfully!")

def main():
    input_file = '../../2022InflectionST/part1/development_languages/pol_pool.train'
    output_file = 'pol_density.tsv'

    print(f"Reading data from {input_file}...")
    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        dataset = [(line.strip().split('\t')[0], line.strip().split('\t')[2].split(';')) for line in lines]

    # Extract lemmas and create a character set
    lemmas = [data[0] for data in dataset]
    features = [data[1] for data in dataset]
    charset = sorted(set(''.join(lemmas)))
    feature_set = sorted(set(feature for feature_list in features for feature in feature_list))
    print(f"Found {len(lemmas)} unique lemmas and {len(feature_set)} unique features.")

    # Compute densities
    avg_lemma_sims, avg_feature_sims, densities = compute_density(dataset, charset, feature_set)

    # Save to file
    save_to_file(lemmas, avg_lemma_sims, avg_feature_sims, densities, output_file)

if __name__ == "__main__":
    main()
