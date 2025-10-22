import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
cols = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "income",
]

data = pd.read_csv(url, names=cols, sep=",", skipinitialspace=True, na_values=["?"], on_bad_lines='skip')
data.dropna(inplace=True)

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

models = [
    ('LR', LogisticRegression(solver='liblinear', multi_class='ovr', max_iter=5000)),
    ('LDA', LinearDiscriminantAnalysis()),
    ('KNN', KNeighborsClassifier()),
    ('CART', DecisionTreeClassifier(random_state=5)),
    ('NB', GaussianNB()),
    ('SVM', SVC(kernel='rbf', gamma='auto'))
]

print("Model Comparison on Adult Income dataset")
results = {}
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=5)
    cv_results = cross_val_score(model, X_train_scaled, y_train, cv=kfold, scoring='accuracy', n_jobs=-1)
    results[name] = cv_results
    print(f"{name}: {cv_results.mean():.4f} (+/- {cv_results.std():.4f})")

df_results = pd.DataFrame(results)
plt.figure(figsize=(10, 7))
sns.boxplot(data=df_results)
plt.title('Model Comparison (Adult Income)', fontsize=16)
plt.xlabel('Algorithm', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.show()
