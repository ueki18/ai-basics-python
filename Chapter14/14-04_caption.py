from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# モデルの読み込み
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

# 画像の読み込み
image = Image.open("sample.jpg").convert("RGB")

# 前処理
inputs = processor(images=image, return_tensors="pt")

# キャプション生成
out = model.generate(**inputs)

# 結果の表示
caption = processor.decode(out[0], skip_special_tokens=True)
print(caption)
