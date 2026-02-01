import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def compute_minkowski_distance(vector_a,vector_b,p):
    return sum(abs(a - b)** p for a,b in zip(vector_a,vector_b))
dataframe=pd.read_csv(
    r"D:\Sem4\bhuvi\BERT_embeddings.csv"
    )
label_column = dataframe.columns[-1]
features=dataframe.drop(columns=[label_column]).values
vector_1 = features[0]
vector_2 = features[1]
p_values = range(1,11)
distances = []
for p in p_values:
    distance = compute_minkowski_distance(vector_1,vector_2,p)
    distances.append(distance)
plt.plot(p_values,distances,marker='o')
plt.xlabel("p value")
plt.ylabel("Minkowski Distance")
plt.show()
print("p values:", list(p_values))
print("Minkowski distances:", distances)
