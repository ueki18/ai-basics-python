import cv2

img = cv2.imread("image.jpg", 0)

sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # 横方向の勾配
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # 縦方向の勾配

cv2.imshow("sobel_x", sobel_x)
cv2.waitKey(0)
