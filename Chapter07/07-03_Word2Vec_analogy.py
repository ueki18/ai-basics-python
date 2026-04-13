from gensim.models import Word2Vec

# 保存したモデルの読み込み
model = Word2Vec.load("text8_word2vec.model")

# ベクトル演算
print("\nWords similar to 'king - man + woman':")
result = model.wv.most_similar(
    positive=["king", "woman"],
    negative=["man"],
    topn=10
)

for word, score in result:
    print(f"{word} (similarity: {score:.3f})")
