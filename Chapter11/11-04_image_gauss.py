import cv2

img = cv2.imread("image.jpg")

blur = cv2.GaussianBlur(img, (5,5), 0)

cv2.imshow("blur", blur)
cv2.waitKey(0)
