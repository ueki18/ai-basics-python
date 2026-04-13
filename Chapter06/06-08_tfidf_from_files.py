import glob
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# テキストファイルの一覧を取得
files = glob.glob("data/*.txt")

documents = []
for file in files:
    with open(file, encoding="utf-8") as f:
        documents.append(f.read())

# TF-IDFの計算
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

print("文書数と特徴量数:", X.shape)

# 小数点3桁で表示
np.set_printoptions(precision=3, suppress=True)

feature_names = vectorizer.get_feature_names_out()
vectors = X.toarray()

# ヘッダ（単語）
print("\nTF-IDF 行列：")
print("      ", end="")
for word in feature_names:
    print(f"{word:>10}", end="")
print()

# 各文書
for i, vec in enumerate(vectors):
    print(f"doc{i+1:02d}", end="")
    for v in vec:
        print(f"{v:10.3f}", end="")
    print()
