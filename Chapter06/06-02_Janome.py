from janome.tokenizer import Tokenizer

t = Tokenizer()

text = "私は昨日面白い映画を見ました"

for token in t.tokenize(text):
    print(token.surface, token.base_form, token.part_of_speech)
