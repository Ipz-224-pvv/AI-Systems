import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

input_file = '/content/drive/MyDrive/Colab Notebooks/data_regr_4.txt'
X = np.array([2, 4, 6, 8, 10, 12]).reshape(-1, 1)
y = np.array([6.5, 4.4, 3.8, 3.5, 3.1, 3.0])

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

plt.figure(figsize=(7,5))
plt.scatter(X, y, color='green', label='Реальні дані')
plt.plot(X, y_pred, color='black', linewidth=2, label='Лінійна регресія')
plt.title("Лінійна регресія однієї змінної (твої дані)")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("Середня абсолютна похибка (MAE):", round(mae, 4))
print("Середньоквадратична похибка (MSE):", round(mse, 4))
print("Коефіцієнт детермінації (R²):", round(r2, 4))

print("\nРівняння регресії виду y = w1*x + b:")
print(f"y = {model.coef_[0]:.4f} * x + ({model.intercept_:.4f})")
