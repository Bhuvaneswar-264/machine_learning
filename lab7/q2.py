import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier

# Load dataset
df = pd.read_csv("BERT_embeddings (4).csv")

X = df.drop("label", axis=1)
y = df["label"]

# Model
model = DecisionTreeClassifier()

# Parameters
params = {
    "max_depth": [3, 5, 10, None],
    "min_samples_split": [2, 5, 10],
    "criterion": ["gini", "entropy"]
}

# Random Search
random_search = RandomizedSearchCV(model, params, n_iter=5, cv=3)
random_search.fit(X, y)

print("Best Parameters:", random_search.best_params_)