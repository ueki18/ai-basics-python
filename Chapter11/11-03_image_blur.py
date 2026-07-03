import cv2

img = cv2.imread("noise.png")

mean_blur = cv2.blur(img, (5,5))

cv2.imshow("blur", blur)
cv2.waitKey(0)
