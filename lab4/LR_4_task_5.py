import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

m = 100
x = np.linspace(-3, 3, m).reshape(-1, 1)
y = 3 + np.sin(x).ravel() + np.random.uniform(-0.5, 0.5, m)

lin_reg = LinearRegression()
lin_reg.fit(x, y)
y_lin_pred = lin_reg.predict(x)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(x)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_poly_pred = poly_reg.predict(X_poly)

mse_lin = mean_squared_error(y, y_lin_pred)
r2_lin = r2_score(y, y_lin_pred)
mse_poly = mean_squared_error(y, y_poly_pred)
r2_poly = r2_score(y, y_poly_pred)

print("=== Лінійна регресія ===")
print(f"Рівняння: y = {lin_reg.coef_[0]:.3f} * x + ({lin_reg.intercept_:.3f})")
print(f"MSE = {mse_lin:.3f}, R² = {r2_lin:.3f}\n")

a1, a2 = poly_reg.coef_
b = poly_reg.intercept_
print("=== Поліноміальна регресія (ступінь 2) ===")
print(f"Рівняння: y = {a2:.3f} * x² + ({a1:.3f}) * x + ({b:.3f})")
print(f"MSE = {mse_poly:.3f}, R² = {r2_poly:.3f}")

x_new = np.linspace(-3, 3, 200).reshape(-1, 1)
y_lin_new = lin_reg.predict(x_new)
y_poly_new = poly_reg.predict(poly.transform(x_new))

plt.figure(figsize=(10,6))
plt.scatter(x, y, color='blue', s=20, alpha=0.6, label='Згенеровані дані')

plt.plot(x_new, y_lin_new, 'r--', lw=2, label='Лінійна регресія')
for xi, yi, ypi in zip(x, y, y_lin_pred):
    plt.plot([xi, xi], [yi, ypi], 'r', alpha=0.3)

plt.plot(x_new, y_poly_new, color='orange', lw=2, label='Поліноміальна регресія (ступінь 2)')
for xi, yi, ypi in zip(x, y, y_poly_pred):
    plt.plot([xi, xi], [yi, ypi], color='orange', alpha=0.3)

plt.title("Порівняння лінійної та поліноміальної регресій з похибкою")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
