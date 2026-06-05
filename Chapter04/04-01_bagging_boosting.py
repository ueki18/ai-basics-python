from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# データ読み込み
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# バギング
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=10
)
bagging.fit(X_train, y_train)

# ブースティング（AdaBoost）
boosting = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=10
)
boosting.fit(X_train, y_train)

print("Bagging accuracy :", bagging.score(X_test, y_test))
print("Boosting accuracy:", boosting.score(X_test, y_test))
