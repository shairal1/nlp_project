import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns


def read_data(file_path):
    """Read the processed data."""
    return pd.read_csv(file_path)

def load_glove_embeddings(glove_file_path):
    """Load GloVe word vectors from file."""
    embeddings = {}
    with open(glove_file_path, encoding="utf8") as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            embeddings[word] = vector
    return embeddings

def get_sentence_vector(sentence, embeddings, vector_size=100):
    """Compute the average word embedding vector for a sentence."""
    words = sentence.split()
    vectors = [embeddings[word] for word in words if word in embeddings]
    if not vectors:
        return np.zeros(vector_size)
    return np.mean(vectors, axis=0)

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0

def plot_similarity_matrix(similarity_matrix):
    """Plot heatmap of similarity matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("Cosine Similarity Between Sentences")
    plt.xlabel("Sentence Index")
    plt.ylabel("Sentence Index")
    plt.tight_layout()
    plt.savefig("matrix_task4/similarity_matrix.png")
    plt.close()

def main():
    # Load data and GloVe embeddings
    print("Loading data and embeddings...")
    df = read_data('processed_data.csv')
    glove_path = 'glove.6B.100d.txt'  # make sure this file is available
    glove = load_glove_embeddings(glove_path)

    # Select 15 random negative sentences
    negative_df = df[df['sentiment'] == 'negative']
    selected = negative_df.sample(n=15, random_state=42)['processed_text'].tolist()

    # Get sentence vectors
    print("Generating sentence vectors...")
    sentence_vectors = [get_sentence_vector(sent, glove, vector_size=100) for sent in selected]

    # Compute similarity matrix
    sim_matrix = np.zeros((15, 15))
    for i in range(15):
        for j in range(15):
            sim_matrix[i, j] = cosine_similarity(sentence_vectors[i], sentence_vectors[j])

    # Plot and save similarity matrix
    plot_similarity_matrix(sim_matrix)

    # Show most similar pairs
    print("\nMost similar sentence pairs:")
    upper_triangle = np.triu(sim_matrix, k=1)
    flat = upper_triangle.flatten()
    top_indices = np.argsort(flat)[-5:][::-1]
    for idx in top_indices:
        i, j = np.unravel_index(idx, sim_matrix.shape)
        print(f"\nPair ({i+1}, {j+1}) - Similarity: {sim_matrix[i, j]:.3f}")
        print(f"Sentence {i+1}: {selected[i]}")
        print(f"Sentence {j+1}: {selected[j]}")

    # Save to file
    with open("similarity_analysis.txt", "w") as f:
        f.write("Selected Negative Sentences:\n")
        for i, s in enumerate(selected, 0):
            f.write(f"{i}. {s}\n")
        f.write("\nTop 5 Most Similar Sentence Pairs:\n")
        for idx in top_indices:
            i, j = np.unravel_index(idx, sim_matrix.shape)
            f.write(f"\nPair ({i}, {j}) - Similarity: {sim_matrix[i, j]:.3f}\n")
            f.write(f"Sentence {i}: {selected[i]}\n")
            f.write(f"Sentence {j}: {selected[j]}\n")

if __name__ == "__main__":
    main()
