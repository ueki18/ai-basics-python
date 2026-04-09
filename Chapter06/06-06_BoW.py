from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "私 は 映画 が とても 好き です 映画 は 素晴らしい",
    "私 は 音楽 が とても 好き です 音楽 は 楽しい"
]

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(documents)

print(vectorizer.get_feature_names_out())
print(X.toarray())
