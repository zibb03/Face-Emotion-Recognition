import numpy as np
import cv2
from scipy.ndimage import label

#https://m.blog.naver.com/PostView.nhn?blogId=samsjang&logNo=220601488606&proxyReferer=https:%2F%2Fwww.google.com%2F
#잔상 없애보기

cap = cv2.VideoCapture('C:/Users/user/PycharmProjects/OpenCV/doc/watershed_TestVideo/fail_3.mp4')
x, y = int(cap.get(3)), int(cap.get(4))
bgd = cv2.imread('C:/Users/user/PycharmProjects/OpenCV/Background/sky1.jpg')
bgd = cv2.resize(bgd, (x, y))
while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret2, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #내가 삽입
    '''kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=2)'''

    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV)
    binary = cv2.bitwise_not(thr)
    frame_bgd_add = cv2.add(frame, bgd, mask=binary)
    a = cv2.add(frame, frame_bgd_add)

    hsv = cv2.cvtColor(a, cv2.COLOR_BGR2HSV)


    lower_white = np.array([1, 1, 1], np.uint8)
    upper_white = np.array([255, 255, 255], np.uint8)
    frame_threshed = cv2.inRange(hsv, lower_white, upper_white)
    kernel = np.ones((5, 5), np.uint8)

    dilation = cv2.dilate(frame_threshed, kernel, iterations=2)
    a[dilation == 0] = (0, 0, 0)
    cv2.imshow('a', a)

    b = cv2.add(a, bgd, mask=binary)
    cv2.imshow('b', b)
    result = cv2.add(a,b)
    cv2.imshow('result',result)

    cv2.imshow("frame2", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()

