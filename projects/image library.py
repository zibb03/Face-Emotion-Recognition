import cv2
import random

x = random.randint(0, 2)
#EMOTIONS = ["Angry", "Disgusting", "Fearful", "Happy", "Sad", "Surpring", "Neutral"]

#감정 변수 emotion으로 사용중
#emotion 값에 따라 background 사진 출력(여러 이미지 중 랜덤으로)


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
img11 = cv2.imread('C:/Users/user/PycharmProjects/OpenCV/sky.jpg')
img12 = cv2.imread('C:/Users/user/PycharmProjects/OpenCV/party.jpg')


Angry= [img1, img2, img3]
Disgusting = [img4, img5, img6]
'''Fearful = [img7, im82, img9]
Happy = [img10, img11, img12]
Sad = [img13, img14, img15]
Surprising = [img16, img17, img18]
Neutral = [img19, img20, img21]'''

cv2.imshow('Angry', Angry[x])
cv2.waitKey(0)