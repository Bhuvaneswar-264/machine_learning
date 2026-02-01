import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
def compute_confusion_matrix(y_true, y_pred):
    TP = TN = FP = FN = 0

    for true, pred in zip(y_true, y_pred):
        if true == 1 and pred == 1:
            TP += 1
        elif true == 0 and pred == 0:
            TN += 1
        elif true == 0 and pred == 1:
            FP += 1
        elif true == 1 and pred == 0:
            FN += 1

    return np.array([[TN, FP], [FN, TP]])


def compute_accuracy(conf_matrix):
    TN, FP, FN, TP = conf_matrix.ravel()
    return (TP + TN) / (TP + TN + FP + FN)


def compute_precision(conf_matrix):
    TN, FP, FN, TP = conf_matrix.ravel()
    return TP / (TP + FP) if (TP + FP) != 0 else 0


def compute_recall(conf_matrix):
    TN, FP, FN, TP = conf_matrix.ravel()
    return TP / (TP + FN) if (TP + FN) != 0 else 0


def compute_fbeta_score(conf_matrix, beta):
    precision = compute_precision(conf_matrix)
    recall = compute_recall(conf_matrix)
    beta_sq = beta ** 2

    numerator = (1 + beta_sq) * precision * recall
    denominator = (beta_sq * precision) + recall

    return numerator / denominator if denominator != 0 else 0
dataframe = pd.read_csv(
    r"D:/Sem4/bhuvi/BERT_embeddings.csv"
)
label_column = dataframe.columns[-1]

X = dataframe.drop(columns=[label_column]).values
y = dataframe[label_column].values
unique_classes = np.unique(y)
y_binary = np.where(y == unique_classes[0], 0, 1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.3, random_state=42
)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
conf_matrix = compute_confusion_matrix(y_test, y_pred)
accuracy = compute_accuracy(conf_matrix)
precision = compute_precision(conf_matrix)
recall = compute_recall(conf_matrix)
f1_score = compute_fbeta_score(conf_matrix, beta=1)
print("Confusion Matrix:\n", conf_matrix)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)

