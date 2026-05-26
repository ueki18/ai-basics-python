from diffusers import StableDiffusionPipeline
import torch

# GPUが利用できる場合は自動的にGPUを使用する
device = "cuda" if torch.cuda.is_available() else "cpu"

# モデルの読み込み（CPUではfloat16非対応のためfloat32を使用）
dtype = torch.float16 if device == "cuda" else torch.float32
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=dtype
)
pipe = pipe.to(device)

# 生成に用いるテキスト
prompt = "A cat sleeping on the bed"

# 画像生成
image = pipe(prompt).images[0]

# 保存
image.save("generated_image.png")
