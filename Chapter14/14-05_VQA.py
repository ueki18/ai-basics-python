from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image

# モデルの読み込み
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-vqa-base"
)
model = BlipForQuestionAnswering.from_pretrained(
    "Salesforce/blip-vqa-base"
)

# 画像の読み込み
image = Image.open("sample.jpg").convert("RGB")

# 質問の設定
question = "What is the dog doing?"

# 前処理
inputs = processor(images=image, text=question, return_tensors="pt")

# 回答の生成
out = model.generate(**inputs)

# 結果の表示
answer = processor.decode(out[0], skip_special_tokens=True)
print("Q:", question)
print("A:", answer)
