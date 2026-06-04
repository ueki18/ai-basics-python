import cv2

img = cv2.imread("blurred.png")

blur = cv2.GaussianBlur(img, (15,15), 3)
sharp = cv2.addWeighted(img, 2.0, blur, -1.0, 0)  # 元画像を強調，ぼかし画像を減算

cv2.imshow("sharp", sharp)
cv2.waitKey(0)
