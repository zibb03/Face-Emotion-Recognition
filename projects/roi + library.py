import cv2
import numpy as np
import random

x = random.randint(0, 2)

img1 = cv2.imread('C:/Users/user/PycharmProjects/OpenCV/people sixteen.jpg')
img2 = cv2.imread('C:/Users/user/PycharmProjects/OpenCV/people three.jpg')
img3 = cv2.imread('C:/Users/user/PycharmProjects/OpenCV/people twenty four.jpg')
img4 = cv2.imread('C:/Users/user/PycharmProjects/OpenCV/peoples.jpg')
img5 = cv2.imread('C:/Users/user/PycharmProjects/OpenCV/person2.jpg')
img6 = cv2.imread('C:/Users/user/PycharmProjects/OpenCV/persontwo.jpg')
img7 = cv2.imread('C:/Users/user/PycharmProjects/OpenCV/persontwo.jpg')
img8 = cv2.imread('C:/Users/user/PycharmProjects/OpenCV/persontwo.jpg')
img9 = cv2.imread('C:/Users/user/PycharmProjects/OpenCV/persontwo.jpg')
img10 = cv2.imread('C:/Users/user/PycharmProjects/OpenCV/park.jpg')
img11 = cv2.imread('C:/Users/user/PycharmProjects/OpenCV/sky1.jpg')
img12 = cv2.imread('C:/Users/user/PycharmProjects/OpenCV/party.jpg')

Angry= [img1, img2, img3]
Disgusting = [img4, img5, img6]
Fearful = [img7, img8, img9]
Happy = [img10, img11, img12]
'''Sad = [img13, img14, img15]
Surprising = [img16, img17, img18]
Neutral = [img19, img20, img21]'''

 # 이미지 지정
background = Happy[x]

'''cv2.resize(원본, dsize=(0, 0),가로배수,세로배수, interpolation=cv2.INTER_LINEAR)'''

logo = cv2.imread("C:/Users/user/PycharmProjects/OpenCV/Black Layer.png")

gray_logo = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
_, mask_inv = cv2.threshold(gray_logo, 10, 255, cv2.THRESH_BINARY_INV)

background_height, background_width, _ = background.shape # 900, 600, 3
logo_height, logo_width, _ = logo.shape # 360, 313, 3

x = background_height - logo_height
y = (background_width - logo_width) // 2

roi = background[x: x+logo_height, y: y+logo_width]
roi_logo = cv2.add(logo, roi, mask=mask_inv)
result = cv2.add(roi_logo, logo)

np.copyto(roi, result)
cv2.imshow("result_background", background)
cv2.waitKey()