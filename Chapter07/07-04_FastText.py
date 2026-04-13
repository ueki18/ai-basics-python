import gensim.downloader as api
from gensim.models import FastText

# データの読み込み
dataset = api.load("text8")

# FastTextモデルの学習
model = FastText(
    dataset,
    vector_size=100,  # 単語ベクトルの次元数
    window=5,         # 前後の文脈サイズ
    min_count=5,      # 最小出現回数
    workers=4,        # 並列処理数
    sg=1              # Skip-gram
)

# モデルの保存
model.save("text8_fasttext.model")

# モデルの読み込み
loaded_model = FastText.load("text8_fasttext.model")

# 類似単語の取得
results = loaded_model.wv.most_similar("dog", topn=10)

print("Words similar to 'dog':")
for word, score in results:
    print(f"{word} (similarity: {score:.3f})")
