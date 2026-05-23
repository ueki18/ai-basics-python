import numpy as np
from sklearn.linear_model import LinearRegression

# サンプルデータ
X = np.array([
    [1, 2],
    [2, 1],
    [3, 4],
    [4, 3],
    [5, 5]
])

# 目的変数
y = np.array([5, 6, 9, 10, 12])

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

print("係数:", model.coef_)
print("切片:", model.intercept_)
