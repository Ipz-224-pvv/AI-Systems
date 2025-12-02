import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = np.loadtxt("/content/drive/MyDrive/Colab Notebooks/lab5/data_random_forests.txt", delimiter=",")
X, y = data[:, :2], data[:, 2].astype(int)

plt.figure(figsize=(7,5))
plt.scatter(X[:,0], X[:,1], c=y, cmap="viridis", s=30)
plt.title("Вихідні дані для класифікації (три класи)")
plt.xlabel("Ознака 1")
plt.ylabel("Ознака 2")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
et = ExtraTreesClassifier(n_estimators=100, random_state=42)

rf.fit(X_train, y_train)
et.fit(X_train, y_train)

acc_rf = accuracy_score(y_test, rf.predict(X_test))
acc_et = accuracy_score(y_test, et.predict(X_test))

print(f"Random Forest accuracy: {acc_rf:.3f}")
print(f"Extra Trees accuracy: {acc_et:.3f}")

def plot_classifier(model, title):
    h = 0.02
    x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
    y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure(figsize=(7,5))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    plt.scatter(X[:,0], X[:,1], c=y, cmap='viridis', edgecolor='k', s=30)
    plt.title(title)
    plt.xlabel("Ознака 1")
    plt.ylabel("Ознака 2")
    plt.show()

plot_classifier(rf, f"Random Forest (accuracy = {acc_rf:.3f})")
plot_classifier(et, f"Extra Trees (accuracy = {acc_et:.3f})")
