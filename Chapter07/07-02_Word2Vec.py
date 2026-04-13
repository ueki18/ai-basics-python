import gensim.downloader as api
from gensim.models import Word2Vec

# データの読み込み
dataset = api.load("text8")

# パラメータを指定して学習
model = Word2Vec(
    dataset,
    vector_size=100,  # 単語ベクトルの次元数
    window=5,         # 前後何単語を文脈として考慮するか
    min_count=5,      # 出現回数が5回未満の単語は無視
    workers=4,        # 並列処理数（CPUコア数に依存）
    sg=1              # Skip-gramを指定
)

# モデルの保存
model.save("text8_word2vec.model")

# モデルの読み込み
loaded_model = Word2Vec.load("text8_word2vec.model")

results = loaded_model.wv.most_similar("dog", topn=10)

print("Words similar to 'dog':")
for word, score in results:
    print(f"{word} (similarity: {score:.3f})")
