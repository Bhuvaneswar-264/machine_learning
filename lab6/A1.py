import pandas as pd
import numpy as np

df = pd.read_csv("BERT_embeddings (4).csv")

def equal_width_binning(data, bins=4):
    min_val = data.min()
    max_val = data.max()
    width = (max_val - min_val) / bins
    return ((data - min_val) / width).astype(int)

def entropy(labels):
    values, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs))

ent = entropy(df['label'])
print("Entropy:", ent)