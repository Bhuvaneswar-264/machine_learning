import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

sil_scores = []
ch_scores = []
db_scores = []
distortions = []

k_values = range(2, 11)

for k in k_values:
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    km.fit(X_cluster)

    sil_scores.append(silhouette_score(X_cluster, km.labels_))
    ch_scores.append(calinski_harabasz_score(X_cluster, km.labels_))
    db_scores.append(davies_bouldin_score(X_cluster, km.labels_))
    distortions.append(km.inertia_)