import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

cols = ["age","workclass","fnlwgt","education","education-num","marital-status",
        "occupation","relationship","race","sex","capital-gain","capital-loss",
        "hours-per-week","native-country","income"]

data = pd.read_csv(url, names=cols, sep=",", skipinitialspace=True, na_values=["?"], on_bad_lines='skip')
data.dropna(inplace=True)

categorical_cols = data.select_dtypes(include="object").columns.drop("income")
data = pd.get_dummies(data, columns=categorical_cols)
data["income"] = data["income"].astype("category").cat.codes

X = data.drop("income", axis=1)
y = data["income"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LinearSVC(max_iter=10000, random_state=0)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Завдання 2.1. Класифікація за допомогою машин опорних векторів(SVM)")
print("-" * 70)
print(classification_report(y_test, y_pred, target_names=["<=50K", ">50K"]))
print("-" * 70)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("-" * 70)