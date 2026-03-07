import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

df = pd.read_csv("BERT_embeddings (4).csv")

X = df.drop("label", axis=1)
y = df["label"]

model = DecisionTreeClassifier(max_depth=3)
model.fit(X, y)

plt.figure(figsize=(15,10))
plot_tree(model, filled=True)
plt.show()