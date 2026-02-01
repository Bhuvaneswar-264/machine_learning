import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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
k_values = range(1, 12)
accuracies = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    acc = knn.score(X_test, y_test)
    accuracies.append(acc)
accuracy_k1 = accuracies[0] 
accuracy_k3 = accuracies[2]
plt.plot(k_values, accuracies, marker='o')
plt.xlabel("k value")
plt.ylabel("Accuracy")
plt.title("Accuracy vs k for kNN Classifier")
plt.show()
print("Accuracy for NN (k = 1):", accuracy_k1)
print("Accuracy for kNN (k = 3):", accuracy_k3)
