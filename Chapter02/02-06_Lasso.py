import numpy as np
from sklearn.linear_model import Lasso

X = np.array([
    [1, 2],
    [2, 1],
    [3, 4],
    [4, 3],
    [5, 5]
])

y = np.array([5, 6, 9, 10, 12])

model = Lasso(alpha=0.1)
model.fit(X, y)

print("係数:", model.coef_)
print("切片:", model.intercept_)
