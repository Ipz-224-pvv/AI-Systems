import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, cohen_kappa_score, matthews_corrcoef,
    confusion_matrix
)

def load_data():
    iris = load_iris()
    X, y = iris.data, iris.target
    target_names = iris.target_names
    return X, y, target_names

def train_model(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    clf = RidgeClassifier(tol=1e-2, solver="sag")
    clf.fit(X_train_scaled, y_train)
    return clf, scaler

def evaluate_model(clf, scaler, X_test, y_test, target_names):
    X_test_scaled = scaler.transform(X_test)
    y_pred = clf.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    kappa = cohen_kappa_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (weighted): {prec:.4f}")
    print(f"Recall (weighted): {rec:.4f}")
    print(f"F1 (weighted): {f1:.4f}")
    print(f"Cohen Kappa: {kappa:.4f}")
    print(f"Matthews Corrcoef: {mcc:.4f}")
    print("\nClassification Report:\n",
          classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
    
    return y_pred

def plot_confusion(y_test, y_pred, target_names, save_path_prefix="Confusion"):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cbar=False, square=True,
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('RidgeClassifier — Confusion Matrix (Iris)')
    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}.jpg", dpi=200)
    plt.savefig(f"{save_path_prefix}.svg")
    plt.show()

def predict_new_sample(clf, scaler, X_new, target_names):
    X_new_scaled = scaler.transform(X_new)
    pred = clf.predict(X_new_scaled)[0]
    print("Prediction for X_new:", target_names[pred])

X, y, target_names = load_data()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y)
clf, scaler = train_model(X_train, y_train)
y_pred = evaluate_model(clf, scaler, X_test, y_test, target_names)
plot_confusion(y_test, y_pred, target_names)
X_new = np.array([[5, 2.9, 1, 0.2]])
predict_new_sample(clf, scaler, X_new, target_names)
