import cv2
import winsound
from matplotlib import pyplot as plt
child = 'C:/Users/user/PycharmProjects/OpenCV/People/people4.jpg'
image = cv2.imread(child, cv2.IMREAD_UNCHANGED)
grayImage = cv2.imread(child, cv2.IMREAD_GRAYSCALE)

eyes_cascade = cv2.CascadeClassifier(
    'C:/Users/user/PycharmProjects/OpenCV/haarcascades/haarcascade_eye.xml')
eyes = eyes_cascade.detectMultiScale(grayImage, 1.01, 10)

face_cascade = cv2.CascadeClassifier(
    'C:/Users/user/PycharmProjects/OpenCV/haarcascades/haarcascade_frontalcatface.xml')
face = face_cascade.detectMultiScale(grayImage, 1.01, 10)

for (x, y, w, h) in face:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)

    face_image_gray = grayImage[y:y + h, x:x + w]
    face_image_color = image[y:y + h, x:x + w]

    faces_in_body = face_cascade.detectMultiScale(face_image_gray)

    eyes_in_faces = eyes_cascade.detectMultiScale(face_image_gray)


    for (xf, yf, wf, hf) in eyes_in_faces:
        print(xf, yf, wf, hf)
        cv2.rectangle(face_image_color, (xf, yf), (xf + wf, yf + hf), (0, 255, 0), 2)
        cv2.putText(image, "Mask X", (0, image.shape[0] - 10),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 0), 1);

    winsound.PlaySound('warning.wav', winsound.SND_FILENAME)
    plt.figure(figsize=(12, 12))
    plt.imshow(image)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
    exit()


for (a, b, c, d) in eyes:
    cv2.rectangle(image, (a, b), (a + c, b + d), (0, 0, 255), 3)

    cv2.putText(image, "Posung Corona Visualization (Mask O)", (0, image.shape[0] - 10),
                cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 0), 1);

plt.figure(figsize=(12, 12))
plt.imshow(image)
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()