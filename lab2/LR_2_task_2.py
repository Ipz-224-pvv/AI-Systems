import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

cols = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "income",
]

data = pd.read_csv(url, names=cols, sep=",", skipinitialspace=True, na_values=["?"], on_bad_lines='skip')
data.dropna(inplace=True)

class_names = data['income'].astype('category').cat.categories.tolist()

for col in data.select_dtypes(include="object").columns:
    data[col] = data[col].astype("category").cat.codes

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=5, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

specs = [
    ("poly", dict(kernel="poly", degree=8, gamma="scale", cache_size=1000)),
    ("rbf", dict(kernel="rbf", gamma="scale", cache_size=1000)),
    ("sigmoid", dict(kernel="sigmoid", gamma="scale", cache_size=1000)),
]

rows = []
print("-" * 60)
print("Завдання 2.2: Порівняння ядер SVM")

for name, params in specs:
    clf = SVC(**params)
    t0 = time.time()
    clf.fit(X_train_scaled, y_train)
    fit_sec = time.time() - t0
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    
    rep_txt = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
    rep = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    
    print("\n" + "-"*60)
    print(f"SVM kernel: {name.upper()} | accuracy={acc:.4f} | fit_time={fit_sec:.2f}s")
    print("-" * 60)
    print(rep_txt)
    
    rows.append({
        "kernel": name,
        "accuracy": acc,
        "precision_class_1": rep.get(class_names[1], {}).get("precision", np.nan),
        "recall_class_1": rep.get(class_names[1], {}).get("recall", np.nan),
        "f1_class_1": rep.get(class_names[1], {}).get("f1-score", np.nan),
        "fit_time_sec": fit_sec
    })

summary = pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)
print("\nЗведена таблиця результатів:")
print(summary)