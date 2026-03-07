import pandas as pd
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("BERT_embeddings (4).csv")

X = df.drop("label", axis=1)
y = df["label"]

model = DecisionTreeClassifier()
model.fit(X, y)

print("Decision Tree Built")