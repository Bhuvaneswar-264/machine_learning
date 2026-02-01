import numpy as np
import pandas as pd
def compute_mean(data):
    return np.mean(data,axis=0)
def compute_variance(data):
    return np.var(data, axis=0)
def compute_standard_deviation(data):
    return np.std(data,axis=0)
def compute_interclass_distance(mean_class_1,mean_class_2):
    return np.linalg.norm(mean_class_1 - mean_class_2)
dataframe=pd.read_csv(
    r"D:\Sem4\bhuvi\BERT_embeddings.csv"
    )
label_column = dataframe.columns[-1]
features=dataframe.drop(columns=[label_column]).values
labels=dataframe[label_column].values
unique_classes = np.unique(labels)
class_1 = features[labels == unique_classes[0]]
class_2 = features[labels == unique_classes[1]]
mean_class_1 = compute_mean(class_1)
mean_class_2 = compute_mean(class_2)
std_class_1 = compute_standard_deviation(class_1)
std_class_2 =   compute_standard_deviation(class_2)
interclass_distance =  compute_interclass_distance(
    mean_class_1,mean_class_2
    )
print("class_1 Mean Vector:",mean_class_1)
print("class_2 Mean Vector:",mean_class_2)
print("class_1 Intraclass Spread:",std_class_1)
print("class_2 Intraclass Spread:",std_class_2)
print("Intraclass Distance:",interclass_distance)
