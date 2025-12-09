import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth

data = np.loadtxt('/content/drive/MyDrive/Colab Notebooks/data_clustering.txt', delimiter=',')
print("Перші 5 рядків даних:\n", data[:5])

bandwidth = estimate_bandwidth(data, quantile=0.2, n_samples=200)
print(f"\nОцінена ширина ядра (bandwidth): {bandwidth:.3f}")

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
labels = ms.fit_predict(data)
cluster_centers = ms.cluster_centers_

n_clusters_ = len(np.unique(labels))
print(f"\nКількість знайдених кластерів: {n_clusters_}")
print("\nКоординати центрів кластерів:")
print(cluster_centers)

plt.figure(figsize=(7, 6))
colors = plt.cm.tab10(np.linspace(0, 1, n_clusters_))
for k, col in zip(range(n_clusters_), colors):
    class_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(data[class_members, 0], data[class_members, 1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)
    plt.plot(cluster_center[0], cluster_center[1], 'X', markerfacecolor=col, markeredgecolor='k', markersize=14)
plt.title("Кластеризація методом зсуву середнього (Mean Shift)")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()
