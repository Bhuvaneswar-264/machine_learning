import matplotlib.pyplot as plt

plt.figure()
plt.plot(k_values, sil_scores)
plt.title("Silhouette Score vs k")
plt.xlabel("k")
plt.ylabel("Silhouette Score")
plt.show()

plt.figure()
plt.plot(k_values, ch_scores)
plt.title("Calinski-Harabasz Score vs k")
plt.xlabel("k")
plt.ylabel("CH Score")
plt.show()

plt.figure()
plt.plot(k_values, db_scores)
plt.title("Davies-Bouldin Index vs k")
plt.xlabel("k")
plt.ylabel("DB Index")
plt.show()

plt.figure()
plt.plot(k_values, distortions)
plt.title("Elbow Plot (Inertia vs k)")
plt.xlabel("k")
plt.ylabel("Inertia")
plt.show()