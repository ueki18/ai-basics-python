from janome.tokenizer import Tokenizer

t = Tokenizer()

text = "私は昨日面白い映画を見ました"

nouns = []

for token in t.tokenize(text):
    if "名詞" in token.part_of_speech:
        nouns.append(token.surface)

print(nouns)
