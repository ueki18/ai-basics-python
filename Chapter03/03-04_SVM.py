from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# model = SVC(kernel="linear")
model = SVC(kernel="poly", degree=3)
# model = SVC(kernel="rbf")

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("予測結果:", y_pred[:10])
print("正解ラベル:", y_test[:10])
