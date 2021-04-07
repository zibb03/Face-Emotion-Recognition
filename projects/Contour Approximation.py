import cv2
import cv2 as cv
import numpy as np

'''img = cv2.imread("C:/Users/user/PycharmProjects/OpenCV/capture.jpg", cv2.IMREAD_COLOR)
img_gray = cv2.bitwise_not(img)'''


img_color = cv.imread('C:/Users/user/PycharmProjects/OpenCV/persontwo.jpg')
img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)
ret, img_binary = cv.threshold(img_gray, 127, 255, 0)
contours, hierarchy = cv.findContours(img_binary, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)


for cnt in contours:
    cv.drawContours(img_color, [cnt], 0, (255, 0, 0), 3)  # blue

cv.imshow("result", img_color)

cv.waitKey(0)


for cnt in contours:

    epsilon = 0.02 * cv.arcLength(cnt, True)
    approx = cv.approxPolyDP(cnt, epsilon, True)
    print( len(approx))

    cv.drawContours(img_color,[approx],0,(0,255,255),5)


cv.imshow("result", img_color)

cv.waitKey(0)