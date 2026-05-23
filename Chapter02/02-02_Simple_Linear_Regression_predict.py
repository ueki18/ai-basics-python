import numpy as np
from sklearn.linear_model import LinearRegression

# サンプルデータ
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 5])

model = LinearRegression()

# 学習
model.fit(X, y)

# 予測
X_new = np.array([[6]])
print("新しいデータの予測:", model.predict(X_new))
