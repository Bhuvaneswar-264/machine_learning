import pandas as pd
import numpy as np

df = pd.read_csv("BERT_embeddings (4).csv")

def gini_index(labels):
    values, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    return 1 - np.sum(probs**2)

gini = gini_index(df['label'])
print("Gini Index:", gini)