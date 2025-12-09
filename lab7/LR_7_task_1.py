import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = np.loadtxt('/content/drive/MyDrive/Colab Notebooks/data_clustering.txt', delimiter=',')
print(data[:5])

plt.figure(figsize=(6, 6))
plt.scatter(data[:, 0], data[:, 1], s=50)
plt.title("Вхідні дані")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()

kmeans = KMeans(n_clusters=5, init="k-means++", n_init=10, random_state=42)
labels = kmeans.fit_predict(data)
centers = kmeans.cluster_centers_

x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(7, 7))
plt.imshow(Z, interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Pastel1, aspect='auto', origin='lower')
plt.scatter(data[:, 0], data[:, 1], c=labels, s=50, cmap='tab10', edgecolor='k')
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, marker='X', label='Центри кластерів')
plt.title("Результат кластеризації методом k-середніх")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

print(centers)
