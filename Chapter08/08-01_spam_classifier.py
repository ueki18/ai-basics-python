import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# データの読み込み（タブ区切り）
df = pd.read_csv("SMSSpamCollection", sep="\t", header=None)
df.columns = ["label", "text"]

# ラベルを数値に変換
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# 学習データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

# テキストを数値ベクトルに変換（TF-IDF）
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# モデルの学習
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# 予測
y_pred = model.predict(X_test_vec)

# 精度評価
print("Accuracy:", accuracy_score(y_test, y_pred))
