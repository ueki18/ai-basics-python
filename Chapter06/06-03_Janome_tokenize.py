from janome.tokenizer import Tokenizer

t = Tokenizer()

text = "私は昨日面白い映画を見ました"

words = [token.surface for token in t.tokenize(text)]

print(words)
