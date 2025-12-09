import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
import pandas as pd

iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42)
labels = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_

plt.figure(figsize=(7, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='Accent', s=50)
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, marker='X', label='Центри кластерів')
plt.title("Кластеризація набору Iris методом k-середніх")
plt.xlabel("Довжина чашолистка")
plt.ylabel("Ширина чашолистка")
plt.legend()
plt.grid(True)
plt.show()

comparison = pd.crosstab(y, labels, rownames=['Реальний клас'], colnames=['Передбачений кластер'])
print("\n=== Порівняння кластерів із реальними класами ===")
print(comparison)
print("\nКоординати центрів кластерів:")
print(centers)
