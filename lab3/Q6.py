import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
dataframe = pd.read_csv(
    r"D:/Sem4/bhuvi/BERT_embeddings.csv"
)

label_column = dataframe.columns[-1]
X = dataframe.drop(columns=[label_column]).values
y = dataframe[label_column].values
unique_classes = np.unique(y)
class_1 = unique_classes[0]
class_2 = unique_classes[1]
mask = (y == class_1) | (y == class_2)
X_binary = X[mask]
y_binary = y[mask]
X_train, X_test, y_train, y_test = train_test_split(
    X_binary, y_binary, test_size=0.3, random_state=42
)
print("Training feature shape:", X_train.shape)
print("Testing feature shape:", X_test.shape)
print("Training labels count:", len(y_train))
print("Testing labels count:", len(y_test))
