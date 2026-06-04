import cv2
import numpy as np

# 100x100の黒い画像を作成（全ピクセルが0）
img = np.zeros((100, 100, 3), dtype=np.uint8)

# 左上の50x50領域を赤色に設定（BGRの順に注意）
img[0:50, 0:50] = [0, 0, 255]

# ファイルに保存
cv2.imwrite("output.jpg", img)
