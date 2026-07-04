import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# サンプルデータ（y = x^2 の関係）
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1, 4, 9, 16, 25])

# 特徴量を2乗まで拡張する
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# 線形回帰と同じ手順で学習する
model = LinearRegression()
model.fit(X_poly, y)

# 予測（x=6 のとき y=36 になるはず）
X_test = poly.transform([[6]])
print("x=6 の予測値:", model.predict(X_test))

# グラフで確認する
X_plot = np.linspace(0, 6, 100).reshape(-1, 1)
y_plot = model.predict(poly.transform(X_plot))

plt.scatter(X, y, color="blue", label="Data points")
plt.plot(X_plot, y_plot, color="red", label="Polynomial regression")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Polynomial Regression")
plt.show()
