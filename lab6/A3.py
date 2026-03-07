import pandas as pd
import numpy as np

df = pd.read_csv("BERT_embeddings (4).csv")

def entropy(y):
    values, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs*np.log2(probs))

def information_gain(data, feature, target):
    total_entropy = entropy(data[target])
    values = data[feature].unique()
    weighted_entropy = 0

    for v in values:
        subset = data[data[feature] == v]
        weighted_entropy += (len(subset)/len(data))*entropy(subset[target])

    return total_entropy - weighted_entropy

for col in df.columns[:-1]:
    df[col] = pd.qcut(df[col], 4, duplicates='drop')

gains = {col: information_gain(df, col, 'label') for col in df.columns[:-1]}
root = max(gains, key=gains.get)

print("Root Feature:", root)