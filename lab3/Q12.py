import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
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
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_train_pred = knn.predict(X_train)
y_test_pred = knn.predict(X_test)
conf_matrix_train = confusion_matrix(y_train, y_train_pred)
conf_matrix_test = confusion_matrix(y_test, y_test_pred)
precision_train = precision_score(y_train, y_train_pred, average='binary')
recall_train = recall_score(y_train, y_train_pred, average='binary')
f1_train = f1_score(y_train, y_train_pred, average='binary')
precision_test = precision_score(y_test, y_test_pred, average='binary')
recall_test = recall_score(y_test, y_test_pred, average='binary')
f1_test = f1_score(y_test, y_test_pred, average='binary')
print("Confusion Matrix (Training Data):\n", conf_matrix_train)
print("Precision (Train):", precision_train)
print("Recall (Train):", recall_train)
print("F1 Score (Train):", f1_train)

print("\nConfusion Matrix (Test Data):\n", conf_matrix_test)
print("Precision (Test):", precision_test)
print("Recall (Test):", recall_test)
print("F1 Score (Test):", f1_test)
