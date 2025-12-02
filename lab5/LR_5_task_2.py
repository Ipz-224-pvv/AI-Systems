import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

data = np.loadtxt("/content/drive/MyDrive/Colab Notebooks/lab5/data_imbalance.txt", delimiter=",")
X, y = data[:, :-1], data[:, -1].astype(int)

plt.figure(figsize=(6,5))
plt.scatter(X[y==0, 0], X[y==0, 1], s=50, c='blue', label='Class 0')
plt.scatter(X[y==1, 0], X[y==1, 1], s=50, c='orange', label='Class 1')
plt.title("Вхідні дані з дисбалансом класів")
plt.xlabel("Ознака 1")
plt.ylabel("Ознака 2")
plt.legend()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

def train_and_show(model, title):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(title)
    print(classification_report(y_test, y_pred))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.title(title)
    plt.show()

clf_unbalanced = ExtraTreesClassifier(n_estimators=100, max_depth=4, random_state=0)
clf_balanced = ExtraTreesClassifier(n_estimators=100, max_depth=4, random_state=0, class_weight='balanced')

train_and_show(clf_unbalanced, "Матриця невідповідностей (без балансування)")
train_and_show(clf_balanced, "Матриця невідповідностей (з балансуванням)")
