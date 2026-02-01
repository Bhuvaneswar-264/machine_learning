import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def compute_mean(feature_vector):
    return np.mean(feature_vector)
def compute_variance(feature_vector):
    return np.var(feature_vector)
def compute_histogram(feature_vector,bins):
    return np.histogram(feature_vector,bins=bins)
dataframe=pd.read_csv(
    r"D:\Sem4\bhuvi\BERT_embeddings.csv"
    )
label_column=dataframe.columns[-1]
features=dataframe.drop(columns=[label_column]).values
feature_index = 0
selected_feature = features[:,feature_index]
hist_values,bin_edges = compute_histogram(selected_feature,bins=20)
feature_mean = compute_mean(selected_feature)
feature_variance = compute_variance(selected_feature)
plt.hist(selected_feature,bins=20)
plt.xlabel("Feature Values")
plt.ylabel("Frequency")
plt.show()
print("Histogram Values:", hist_values)
print("Bin Edges:", bin_edges)
print("Feature Mean:", feature_mean)
print("Feature Variance:", feature_variance)

