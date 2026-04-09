from janome.tokenizer import Tokenizer

t = Tokenizer()

text = "私は昨日面白い映画を見ました"

base_words = []

for token in t.tokenize(text):
    base_words.append(token.base_form)

print(base_words)
