import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_train, y_train)
test_predictions = knn_classifier.predict(X_test)
test_vector = X_test[0]
single_prediction = knn_classifier.predict([test_vector])
print("Predicted classes for test set:", test_predictions[:10])
print("Actual classes for test set:", y_test[:10])
print("Predicted class for one test vector:", single_prediction[0])
print("Actual class for that test vector:", y_test[0])
