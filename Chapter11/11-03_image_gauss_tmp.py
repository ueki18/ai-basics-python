import cv2

img = cv2.imread("noise.png")

blur = cv2.GaussianBlur(img, (5,5), 0)

cv2.imwrite("out222.png", blur)
