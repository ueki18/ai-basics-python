import torch
import clip
from PIL import Image

# モデルの読み込み
model, preprocess = clip.load("ViT-B/32")

# 画像の読み込み
image = preprocess(Image.open("sample.jpg")).unsqueeze(0)

# テキスト（ラベル候補）
text = clip.tokenize(["a dog", "a cat", "a car"])

# 推論
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    # 類似度計算
    similarity = (image_features @ text_features.T).softmax(dim=-1)

print(similarity)
