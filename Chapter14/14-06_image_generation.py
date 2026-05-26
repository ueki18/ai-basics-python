from diffusers import StableDiffusionPipeline
import torch

# モデルの読み込み
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)

# GPUが利用できる場合は自動的にGPUを使用する
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)

# 生成に用いるテキスト
prompt = "A cat sleeping on the bed"

# 画像生成
image = pipe(prompt).images[0]

# 保存
image.save("generated_image.png")
