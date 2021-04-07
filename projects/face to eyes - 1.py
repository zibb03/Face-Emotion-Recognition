import cv2
import numpy as np

child = 'C:/Users/user/PycharmProjects/OpenCV/People/full body3.jpg'
frame = cv2.imread('C:/Users/user/PycharmProjects/OpenCV/People/full body3.jpg')
image = cv2.imread(child, cv2.IMREAD_UNCHANGED)
grayImage = cv2.imread(child, cv2.IMREAD_GRAYSCALE)

fullbody_cascade = cv2.CascadeClassifier(
    'C:/Users/user/PycharmProjects/OpenCV/pedestrians/haarcascade_fullbody.xml')
fullbody = fullbody_cascade.detectMultiScale(grayImage, 1.01, 10)
hog_cascade = cv2.CascadeClassifier(
    'C:/Users/user/PycharmProjects/OpenCV/pedestrians/hogcascade_pedestrians.xml')
hog = hog_cascade.detectMultiScale(grayImage, 1.01, 10)

#haarcascades
'''for (x, y, w, h) in face:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)

    face_image_gray = grayImage[y:y + h, x:x + w]
    face_image_color = image[y:y + h, x:x + w]

    faces_in_body = face_cascade.detectMultiScale(face_image_gray)

    eyes_in_faces = eye_cascade.detectMultiScale(face_image_gray)'''

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

cv2.imshow('output', output)
cv2.waitKey(0)
cv2.destroyAllWindows()
