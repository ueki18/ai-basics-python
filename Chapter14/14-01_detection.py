from ultralytics import YOLO

# YOLO26モデルの読み込み（事前学習済み）
model = YOLO("yolo26n.pt")

# 画像に対して物体検出を実行
results = model("sample.jpg")

# 結果の表示
results[0].show()

# 検出結果の保存
results[0].save("result.jpg")
