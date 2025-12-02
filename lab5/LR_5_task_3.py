!pip install -q scikit-learn matplotlib numpy
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import ExtraTreesClassifier

data = np.loadtxt("/content/drive/MyDrive/Colab Notebooks/lab5/data_random_forests.txt", delimiter=",")
X, y = data[:, :-1], data[:, -1].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

parameter_grid = [
    {'n_estimators': [100], 'max_depth': [2, 4, 7, 12, 17]},
    {'max_depth': [4], 'n_estimators': [25, 50, 100, 250]}
]

metrics = ['precision_weighted', 'recall_weighted']

for metric in metrics:
    print(f"\n#### Пошук оптимальних параметрів для метрики: {metric}")
    classifier = GridSearchCV(
        ExtraTreesClassifier(random_state=0),
        parameter_grid,
        cv=5,
        scoring=metric
    )
    classifier.fit(X_train, y_train)

    print("\nРезультати перевірки для кожної комбінації параметрів:")
    for params, score in zip(classifier.cv_results_['params'], classifier.cv_results_['mean_test_score']):
        print(f"{params} --> середній бал: {score:.3f}")

    print("\nНайкращі параметри:", classifier.best_params_)

y_pred = classifier.predict(X_test)
print("\n=== Звіт про класифікацію ===")
print(classification_report(y_test, y_pred))
