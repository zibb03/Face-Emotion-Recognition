import cv2
import numpy as np


 # 이미지 지정
background = cv2.imread("C:/Users/user/PycharmProjects/OpenCV/Background/Happy1.jpg")
logo = cv2.imread("C:/Users/user/PycharmProjects/OpenCV/People/upper body5.jpg")

gray_logo = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
_, mask_inv = cv2.threshold(gray_logo, 10, 255, cv2.THRESH_BINARY_INV)

cv2.imshow("mask_inv", mask_inv)
cv2.waitKey()

background_height, background_width, _ = background.shape # 900, 600, 3
logo_height, logo_width, _ = logo.shape # 360, 313, 3

x = background_height - logo_height
'''900 - 360 = 540'''
y = (background_width - logo_width) // 2
'''600 - 313 = 287'''

roi = background[x: x+logo_height, y: y+logo_width]
'''540 + 360, 287 + 313 // 세로, 가로'''
cv2.imshow("ROI", roi)
cv2.waitKey()

roi_logo = cv2.add(logo, roi, mask=mask_inv)
cv2.imshow("roi_logo", roi_logo)
cv2.waitKey()

result = cv2.add(roi_logo, logo)
cv2.imshow("result", result)
cv2.waitKey()

np.copyto(roi, result)
cv2.imshow("result_background", background)
cv2.waitKey()