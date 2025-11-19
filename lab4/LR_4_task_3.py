!pip install -q scikit-learn matplotlib numpy

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

input_file = '/content/drive/MyDrive/Colab Notebooks/lab4/data_multivar_regr.txt'
data = np.loadtxt(input_file, delimiter=',')

X = data[:, :-1]
y = data[:, -1]

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

print("=== Рівняння множинної регресії ===")
terms = " + ".join([f"({coef:.4f})*x{i+1}" for i, coef in enumerate(model.coef_)])
print(f"y = {terms} + ({model.intercept_:.4f})")

mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"\nMSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")

plt.figure(figsize=(8,5))
plt.bar([f'x{i+1}' for i in range(X.shape[1])], model.coef_, alpha=0.7)
plt.xlabel("Ознака")
plt.ylabel("Коефіцієнт")
plt.title("Вплив ознак на результат")
plt.grid(alpha=0.3)
plt.show()
