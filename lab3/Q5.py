import numpy as np
import pandas as pd
from scipy.spatial.distance import minkowski

def compute_minkowski_distance(vector_a, vector_b, p):
    return sum(abs(a - b) ** p for a, b in zip(vector_a, vector_b)) ** (1 / p)
dataframe = pd.read_csv(
    r"D:/Sem4/bhuvi/BERT_embeddings.csv"
)
label_column = dataframe.columns[-1]
features = dataframe.drop(columns=[label_column]).values
vector_1 = features[0]
vector_2 = features[1]
p_value = 2
own_distance = compute_minkowski_distance(vector_1, vector_2, p_value)
scipy_distance = minkowski(vector_1, vector_2, p_value)
print("Minkowski Distance (Own Function):", own_distance)
print("Minkowski Distance (SciPy Function):", scipy_distance)
