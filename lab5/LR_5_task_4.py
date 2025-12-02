!pip install -q scikit-learn matplotlib numpy

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, explained_variance_score

housing = datasets.fetch_california_housing()
X, y = shuffle(housing.data, housing.target, random_state=7)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

regressor = AdaBoostRegressor(
    DecisionTreeRegressor(max_depth=4),
    n_estimators=400,
    random_state=7
)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)

print("РЕЗУЛЬТАТИ")
print(f"Середньоквадратична похибка: {mse:.3f}")
print(f"Пояснена дисперсія: {evs:.3f}")

feature_importances = 100.0 * (regressor.feature_importances_ / max(regressor.feature_importances_))
sorted_idx = np.argsort(feature_importances)
positions = np.arange(len(sorted_idx)) + 0.5

plt.figure(figsize=(10, 6))
plt.barh(positions, feature_importances[sorted_idx], align='center')
plt.yticks(positions, np.array(housing.feature_names)[sorted_idx])
plt.xlabel('Відносна важливість (%)')
plt.title('Важливість ознак (AdaBoostRegressor)')
plt.show()
