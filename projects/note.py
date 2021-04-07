import numpy as np
import cv2
from matplotlib import pyplot as plt
child = 'C:/Users/user/PycharmProjects/OpenCV/clean.jpg'


'''face_cascade = cv2.CascadeClassifier(
    './data/haarcascades/haarcascade_frontalface_default.xml')'''
face_cascade = cv2.CascadeClassifier(
    'C:/Users/user/PycharmProjects/OpenCV/haarcascades/haarcascade_frontalface_default.xml')

grayImage = cv2.imread(child, cv2.IMREAD_GRAYSCALE)
'''cv2.imshow('Gray', image)
cv2.waitKey(0)'''

faces = face_cascade.detectMultiScale(grayImage, 1.03, 5)

print(faces.shape)
print("Number of faces detected: " + str(faces.shape[0]))

'''face_cascade = cv2.CascadeClassifier(
    'C:/Users/user/PycharmProjects/OpenCV/haarcascades/haarcascade_frontalface_default.xml')

people = 'C:/Users/user/PycharmProjects/OpenCV.people.jpg'
image = cv2.imread(people, cv2.IMREAD_UNCHANGED)
image = cv2.imread('C:/Users/user/PycharmProjects/OpenCV.people.jpg')
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(12,8))
plt.imshow(grayImage, cmap='gray')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()'''
