from gensim.models import Word2Vec
import numpy as np

# モデルの読み込み
model = Word2Vec.load("text8_word2vec.model")

def document_vector(text):
    words = text.split()
    vectors = []

    for word in words:
        if word in model.wv:
            vectors.append(model.wv[word])

    if len(vectors) == 0:
        return np.zeros(model.vector_size)

    return np.mean(vectors, axis=0)

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (
        np.linalg.norm(vec1) * np.linalg.norm(vec2)
    )

# 比較する2つの文
text1 = "I like machine learning"
text2 = "I enjoy studying AI"

vec1 = document_vector(text1)
vec2 = document_vector(text2)

similarity = cosine_similarity(vec1, vec2)

print(f"Similarity: {similarity:.3f}")
