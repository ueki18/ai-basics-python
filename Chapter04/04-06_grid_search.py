from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# データ
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=0
)

# パラメータ候補
params = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10]
}

# Grid Search（5-fold交差検証）
grid = GridSearchCV(RandomForestClassifier(), params, cv=5)
grid.fit(X_train, y_train)

# 最適パラメータと検証精度
print("最適パラメータ:", grid.best_params_)
print("検証精度　　　:", grid.best_score_)

# テストデータで最終評価
print("テスト精度　　:", grid.score(X_test, y_test))
