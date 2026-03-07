import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("BERT_embeddings (4).csv")

X = df[[df.columns[0], df.columns[1]]]
y = df["label"]

model = DecisionTreeClassifier()
model.fit(X,y)

x_min, x_max = X.iloc[:,0].min()-1, X.iloc[:,0].max()+1
y_min, y_max = X.iloc[:,1].min()-1, X.iloc[:,1].max()+1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z)
plt.scatter(X.iloc[:,0], X.iloc[:,1], c=y)
plt.show()