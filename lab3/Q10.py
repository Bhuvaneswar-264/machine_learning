import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
def euclidean_distance(vec1, vec2):
    return np.sqrt(np.sum((vec1 - vec2) ** 2))
def own_knn_predict(X_train, y_train, test_vector, k):
    distances = []
    for x, label in zip(X_train, y_train):
        dist = euclidean_distance(x, test_vector)
        distances.append((dist, label))
    distances.sort(key=lambda x: x[0])
    nearest_labels = [label for _, label in distances[:k]]
    return max(set(nearest_labels), key=nearest_labels.count)
def own_knn_accuracy(X_train, y_train, X_test, y_test, k):
    correct = 0
    for x, y_true in zip(X_test, y_test):
        if own_knn_predict(X_train, y_train, x, k) == y_true:
            correct += 1
    return correct / len(y_test)
dataframe = pd.read_csv(
    r"D:/Sem4/bhuvi/BERT_embeddings.csv"
)
label_column = dataframe.columns[-1]
X = dataframe.drop(columns=[label_column]).values
y = dataframe[label_column].values
unique_classes = np.unique(y)
mask = (y == unique_classes[0]) | (y == unique_classes[1])
X = X[mask]
y = y[mask]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
k = 3
sklearn_knn = KNeighborsClassifier(n_neighbors=k)
sklearn_knn.fit(X_train, y_train)
sklearn_accuracy = sklearn_knn.score(X_test, y_test)
own_accuracy = own_knn_accuracy(X_train, y_train, X_test, y_test, k)
print("sklearn kNN Accuracy (k=3):", sklearn_accuracy)
print("Own kNN Accuracy (k=3):", own_accuracy)
