!pip install -q scikit-learn matplotlib numpy

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import sklearn.metrics as sm
import pickle

input_file = '/content/drive/MyDrive/ColabNotebooks/data_singlevar_regr.txt'

X = np.array([2, 4, 6, 8, 10, 12]).reshape(-1, 1)
y = np.array([6.5, 4.4, 3.8, 3.5, 3.1, 3.0])

num_training = int(0.8 * len(X))
num_test = len(X) - num_training

X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]

regressor = linear_model.LinearRegression()
regressor.fit(X_train, y_train)

y_test_pred = regressor.predict(X_test)

plt.figure(figsize=(7,5))
plt.scatter(X_test, y_test, label='Реальні дані')
plt.plot(X_test, y_test_pred, linewidth=3, label='Лінійна регресія')
plt.title("Лінійна регресія однієї змінної")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

print("Linear regressor performance:")
print("Середня абсолютна похибка (MAE) =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Середньоквадратична похибка (MSE) =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Медіанна абсолютна похибка =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Пояснена дисперсія =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("Коефіцієнт детермінації (R²) =", round(sm.r2_score(y_test, y_test_pred), 2))

with open("model.pkl", "wb") as f:
    pickle.dump(regressor, f)

with open("model.pkl", "rb") as f:
    regressor_model = pickle.load(f)

y_test_pred_new = regressor_model.predict(X_test)

print("\nNew mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred_new), 2))

print("Рівняння регресії виду y = w1*x + b:")
print(f"y = {regressor.coef_[0]:.4f} * x + ({regressor.intercept_:.4f})")
