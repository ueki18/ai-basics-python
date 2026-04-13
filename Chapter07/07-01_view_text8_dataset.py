import gensim.downloader as api

dataset = api.load("text8")

# 先頭の単語列を取得
first_data = next(iter(dataset))

# 最初の300単語を表示
print(first_data[:300])
