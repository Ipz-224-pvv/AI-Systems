!pip install -q scikit-learn matplotlib numpy

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

diabetes = datasets.load_diabetes()
X, y = diabetes.data, diabetes.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

y_pred = regr.predict(X_test)

print("=== Коефіцієнти регресії ===")
print(f"Коефіцієнти: {regr.coef_}")
print(f"Вільний член (intercept): {regr.intercept_:.3f}")

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n=== Метрики точності ===")
print(f"MSE: {mse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.4f}")

plt.figure(figsize=(7,5))
plt.scatter(y_test, y_pred, edgecolors='black', color='skyblue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Реальні значення')
plt.ylabel('Передбачені значення')
plt.title('Регресія багатьох змінних')
plt.grid(alpha=0.3)
plt.show()
