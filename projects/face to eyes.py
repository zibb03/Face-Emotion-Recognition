'''import cv2
import numpy as np

child = 'C:/Users/user/PycharmProjects/OpenCV/People/full body3.jpg'
frame = cv2.imread('C:/Users/user/PycharmProjects/OpenCV/People/full body3.jpg')
image = cv2.imread(child, cv2.IMREAD_UNCHANGED)
grayImage = cv2.imread(child, cv2.IMREAD_GRAYSCALE)

face_cascade = cv2.CascadeClassifier(
    'C:/Users/user/PycharmProjects/OpenCV/haarcascades/haarcascade_upperbody.xml')
face = face_cascade.detectMultiScale(grayImage, 1.01, 10)
eye_cascade = cv2.CascadeClassifier(
    'C:/Users/user/PycharmProjects/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
eye = eye_cascade.detectMultiScale(grayImage, 1.01, 10)

#haarcascades
for (x, y, w, h) in face:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)

    face_image_gray = grayImage[y:y + h, x:x + w]
    face_image_color = image[y:y + h, x:x + w]

    faces_in_body = face_cascade.detectMultiScale(face_image_gray)

    eyes_in_faces = eye_cascade.detectMultiScale(face_image_gray)
    for (xf, yf, wf, hf) in eyes_in_faces:
        cv2.rectangle(face_image_color, (xf, yf), (xf + wf, yf + hf), (0, 255, 0), 2)

#grabcut
if (x, y, w, h) is not None:
    for (x, y, w, h) in face:
        rect = (x, int((y - y / 2)), w, h)
        img = frame.copy()
        img2 = img.copy()
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        output = np.zeros(img.shape, np.uint8)

        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        cv2.grabCut(img2, mask, rect, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')
        output = cv2.bitwise_and(img2, img2, mask=mask2)

for (a, b, c, d) in face:
    cv2.rectangle(output, (a, b), (a + c, b + d), (0, 0, 255), 3)

    face_image_gray = grayImage[b:b + d, a:a + c]
    face_image_color = image[b:b + d, a:a + c]

    faces_in_body = face_cascade.detectMultiScale(face_image_gray)

    eyes_in_faces = eye_cascade.detectMultiScale(face_image_gray)
    for (aa, bb, cc, dd) in eyes_in_faces:
        cv2.rectangle(face_image_color, (aa, bb), (aa + cc, bb + dd), (0, 255, 0), 2)

#grabcut
if (a, b, c, d) is not None:
    for (a, b, c, d) in face:
        rect = (a, int((b - b / 2)), c, d)
        img = frame.copy()
        img2 = img.copy()
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        output = np.zeros(img.shape, np.uint8)

        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        cv2.grabCut(img2, mask, rect, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')
        output = cv2.bitwise_and(img2, img2, mask=mask2)

cv2.imshow('output', output)
cv2.waitKey(0)
cv2.destroyAllWindows()'''
'''
# Perform emotion recognition only when face is detected
if len(dets) > 0:
    # Resize the image to 48x48 for neural network
    for k, d in enumerate(dets): # k = 얼굴 인식 갯수, d = 좌표(?)
        fX = d.left()
        fY = d.top()
        fW = d.right()
        fH = d.bottom()
        print(fX, fY, fW, fH)

        # grabcut
        rect = (fX, int((fY - fY / 2)), fW, fH)
        img = frame.copy()
        img2 = img.copy()
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        output = np.zeros(img.shape, np.uint8)

        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        cv2.grabCut(img2, mask, rect, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')
        output = cv2.bitwise_and(img2, img2, mask=mask2)'''


import cv2
from matplotlib import pyplot as plt

child = 'C:/Users/user/PycharmProjects/OpenCV/People/people1.jpg'
image = cv2.imread(child, cv2.IMREAD_UNCHANGED)
grayImage = cv2.imread(child, cv2.IMREAD_GRAYSCALE)

face_cascade = cv2.CascadeClassifier(
    'C:/Users/user/PycharmProjects/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
face = face_cascade.detectMultiScale(grayImage, 1.01, 10)
eye_cascade = cv2.CascadeClassifier(
    'C:/Users/user/PycharmProjects/OpenCV/haarcascades/haarcascade_eye.xml')
eye = eye_cascade.detectMultiScale(grayImage, 1.01, 10)

for (x, y, w, h) in face:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)

    face_image_gray = grayImage[y:y + h, x:x + w]
    face_image_color = image[y:y + h, x:x + w]

    faces_in_body = face_cascade.detectMultiScale(face_image_gray)

    eyes_in_faces = eye_cascade.detectMultiScale(face_image_gray)
    for (xf, yf, wf, hf) in eyes_in_faces:
        print(xf, yf, wf, hf)
        cv2.rectangle(face_image_color, (xf, yf), (xf + wf, yf + hf), (0, 255, 0), 2)

plt.figure(figsize=(12, 12))
plt.imshow(image)
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()