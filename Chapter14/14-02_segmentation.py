from ultralytics import YOLO

# YOLO26セグメンテーションモデルの読み込み（自動ダウンロード）
model = YOLO("yolo26n-seg.pt")

# 画像に対してセグメンテーションを実行
results = model("sample.jpg")

# 結果の表示
results[0].show()

# 結果の保存
results[0].save("result_seg.jpg")
