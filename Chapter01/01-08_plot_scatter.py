import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# マーカーの種類
markers = ["o", "s", "^"]

# 0,1,2 のラベルごとに処理
for i in range(3):
    subset = df[iris.target == i]
    
    plt.scatter(
        subset["sepal length (cm)"],
        subset["petal length (cm)"],
        label=iris.target_names[i],  # 名前はここで使う
        marker=markers[i]
    )

plt.xlabel("sepal length")
plt.ylabel("petal length")
plt.legend()
plt.show()
