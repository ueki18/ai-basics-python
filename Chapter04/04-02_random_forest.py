from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# データ読み込み
iris = load_iris()
X = iris.data
y = iris.target

# データ分割（再現性のためrandom_stateを固定）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# モデル（少しチューニング）
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    random_state=0
)

# 学習
model.fit(X_train, y_train)

# 予測
y_pred = model.predict(X_test)

# 精度評価
acc = accuracy_score(y_test, y_pred)

print("Accuracy :", acc)
