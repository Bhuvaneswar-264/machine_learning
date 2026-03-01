from sklearn.cluster import KMeans

def perform_kmeans(X_train, k, seed=42):
    kmeans = KMeans(
        n_clusters=k,
        random_state=seed,
        n_init="auto"
    )
    kmeans.fit(X_train)
    return kmeans