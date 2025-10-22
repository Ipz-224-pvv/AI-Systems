import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
target_names = iris.target_names

df = pd.DataFrame(X, columns=feature_names)
df["class"] = pd.Categorical.from_codes(y, target_names)

print("-" * 85)
print("Завдання 2.3. Порівняння якості класифікаторів на прикладі класифікації сортів ірисів")
print("-" * 85)
print("\n")
print("Data shape:", df.shape)
print("\nFirst rows:\n", df.head())

plt.figure(figsize=(10,6))
df[feature_names].plot(kind="box", subplots=True, layout=(2,2), figsize=(10,6), sharex=False, sharey=False)
plt.suptitle("Boxplot of Iris Features")
plt.tight_layout()
plt.show()

sns.pairplot(df, hue="class", diag_kind="hist")
plt.suptitle("Pairplot of Iris Dataset", y=1.02)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = [
    ("LR", LogisticRegression(max_iter=200, solver="liblinear")),
    ("LDA", LinearDiscriminantAnalysis()),
    ("KNN", KNeighborsClassifier()),
    ("CART", DecisionTreeClassifier(random_state=42)),
    ("NB", GaussianNB()),
    ("SVM", SVC(gamma="scale", random_state=42))
]

results = {}
print("\nModel Comparison on Iris Dataset")
for name, model in models:
    X_eval_train = X_train_scaled if name in ["KNN", "SVM"] else X_train
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_results = cross_val_score(model, X_eval_train, y_train, cv=kfold, scoring="accuracy")
    results[name] = cv_results
    print(f"{name}: {cv_results.mean():.3f} (+/- {cv_results.std():.3f})")

df_results = pd.DataFrame(results)
plt.figure(figsize=(10,6))
sns.boxplot(data=df_results)
plt.title("Model Comparison (10-fold CV Accuracy)")
plt.ylabel("Accuracy")
plt.grid(True, alpha=0.3)
plt.show()

best_model = SVC(gamma="scale", random_state=42)
best_model.fit(X_train_scaled, y_train)
y_pred = best_model.predict(X_test_scaled)

print("\nBest Model (SVM) Evaluation on Test Set")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=target_names))

X_new = np.array([[5, 2.9, 1, 0.2]])
X_new_scaled = scaler.transform(X_new)
pred_new = best_model.predict(X_new_scaled)[0]
print("Predicted label for new sample:", target_names[pred_new])
