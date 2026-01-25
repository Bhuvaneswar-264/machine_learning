import numpy as np

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def main():
    v1 = np.array([1, 2, 3, 4])
    v2 = np.array([2, 3, 4, 5])

    print("Cosine Similarity:", cosine_similarity(v1, v2))

main()
