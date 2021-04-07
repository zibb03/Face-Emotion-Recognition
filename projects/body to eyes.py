import numpy as np
import cv2
from matplotlib import pyplot as plt
child = 'C:/Users/user/PycharmProjects/OpenCV/persontwo.jpg'
image = cv2.imread(child, cv2.IMREAD_UNCHANGED)
grayImage = cv2.imread(child, cv2.IMREAD_GRAYSCALE)

face_cascade = cv2.CascadeClassifier(
    'C:/Users/user/PycharmProjects/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
face = face_cascade.detectMultiScale(grayImage, 1.01, 10)
body_cascade = cv2.CascadeClassifier(
    'C:/Users/user/PycharmProjects/OpenCV/haarcascades/haarcascade_fullbody.xml')
body = body_cascade.detectMultiScale(grayImage, 1.01, 10)
eye_cascade = cv2.CascadeClassifier(
    'C:/Users/user/PycharmProjects/OpenCV/haarcascades/haarcascade_eye.xml')
eye = eye_cascade.detectMultiScale(grayImage, 1.01, 10)

for (x, y, w, h) in body:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)

    body_image_gray = grayImage[y:y + h, x:x + w]
    body_image_color = image[y:y + h, x:x + w]

    faces_in_body = face_cascade.detectMultiScale(body_image_gray)

    for (xf, yf, wf, hf) in faces_in_body:
        cv2.rectangle(body_image_color, (xf, yf), (xf + wf, yf + hf), (0, 255, 0), 2)

        face_image_gray = grayImage[y:y + h, x:x + w]
        face_image_color = image[y:y + h, x:x + w]

        eyes_in_faces = eye_cascade.detectMultiScale(face_image_gray)
        for (xf, yf, wf, hf) in eyes_in_faces:
            cv2.rectangle(face_image_color, (xf, yf), (xf + wf, yf + hf), (0, 255, 0), 2)

plt.figure(figsize=(12, 12))
plt.imshow(image)
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()