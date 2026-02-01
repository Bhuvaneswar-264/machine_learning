import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
def train_matrix_inversion(X_train, y_train):
    X_bias = np.c_[np.ones(X_train.shape[0]), X_train]   
    W = np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ y_train
    return W
def predict_matrix_inversion(X_test, W):
    X_bias = np.c_[np.ones(X_test.shape[0]), X_test]
    y_pred = X_bias @ W
    return np.where(y_pred >= 0.5, 1, 0)
def compute_accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

dataframe = pd.read_csv(
    r"D:/Sem4/bhuvi/BERT_embeddings.csv"
)
label_column = dataframe.columns[-1]
X = dataframe.drop(columns=[label_column]).values
y = dataframe[label_column].values
classes = np.unique(y)
y_binary = np.where(y == classes[0], 0, 1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.3, random_state=42
)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
knn_accuracy = compute_accuracy(y_test, knn_pred)
W = train_matrix_inversion(X_train, y_train)
matrix_pred = predict_matrix_inversion(X_test, W)
matrix_accuracy = compute_accuracy(y_test, matrix_pred)
print("kNN Accuracy (k = 3):", knn_accuracy)
print("Matrix Inversion Accuracy:", matrix_accuracy)
