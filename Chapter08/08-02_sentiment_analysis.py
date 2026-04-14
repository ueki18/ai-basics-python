from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# データ読み込み
dataset = load_dataset("imdb")

# 学習用・テスト用データを取り出してシャッフル
train_data = dataset["train"].shuffle(seed=42)
test_data = dataset["test"].shuffle(seed=42)

X_train = train_data["text"][:2000]
y_train = train_data["label"][:2000]

X_test = test_data["text"][:1000]
y_test = test_data["label"][:1000]

# ベクトル化
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# モデル
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# 評価
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
