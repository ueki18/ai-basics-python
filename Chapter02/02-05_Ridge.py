import numpy as np
from sklearn.linear_model import Ridge

X = np.array([
    [1, 2],
    [2, 1],
    [3, 4],
    [4, 3],
    [5, 5]
])

y = np.array([5, 6, 9, 10, 12])

model = Ridge(alpha=1.0)
model.fit(X, y)

print("係数:", model.coef_)
print("切片:", model.intercept_)
