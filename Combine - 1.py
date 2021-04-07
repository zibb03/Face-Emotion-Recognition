import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import dlib
import random

# dlib를 상용 하는 이유는 XML파일을 이용하는 것 보다 dlib를 사용하여 인식 하는 것이 더 인식이 잘되는 듯 함
# dlib를 설치하기 위해서는 Anaconda에서 pip install cmake -> pip install dlib 설치

# dlib얼굴 인식 모델, 감정인식 모델 불러오기
#face_detection = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml') #사용 안함
detector = dlib.get_frontal_face_detector() #얼굴 인식을 위한 dlib 기본 안면인식 모델
emotion_classifier = load_model('C:/Users/user/PycharmProjects/OpenCV/emotion_model.hdf5', compile=False)
EMOTIONS = ["Angry", "Disgusting", "Fearful", "Happy", "Sad", "Surpring", "Neutral"]

# 이미지 불러오기
frame = cv2.imread('C:/Users/user/PycharmProjects/OpenCV/People/')

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
canvas = np.zeros((600, 500, 3), dtype="uint8")
dets = detector(frame, 1) #얼굴인식 완료

# haarcascades
child = 'C:/Users/user/PycharmProjects/OpenCV/People/'
image = cv2.imread(child, cv2.IMREAD_UNCHANGED)
grayImage = cv2.imread(child, cv2.IMREAD_GRAYSCALE)

face_cascade = cv2.CascadeClassifier(
    'C:/Users/user/PycharmProjects/OpenCV/haarcascades/haarcascade_upperbody.xml')
face = face_cascade.detectMultiScale(grayImage, 1.01, 10)
eye_cascade = cv2.CascadeClassifier(
    'C:/Users/user/PycharmProjects/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
eye = eye_cascade.detectMultiScale(grayImage, 1.01, 10)

for (a, b, c, d) in face:
    cv2.rectangle(image, (a, b), (a + c, b + d), (0, 0, 255), 3)

    face_image_gray = grayImage[b:b + d, a:a + c]
    face_image_color = image[b:b + d, a:a + c]

    faces_in_body = face_cascade.detectMultiScale(face_image_gray)

    eyes_in_faces = eye_cascade.detectMultiScale(face_image_gray)
    for (xf, yf, wf, hf) in eyes_in_faces:
        cv2.rectangle(face_image_color, (xf, yf), (xf + wf, yf + hf), (0, 255, 0), 2)
        # --haarcascades

# Perform emotion recognition only when face is detected
if len(dets) > 0:
    # Resize the image to 48x48 for neural network
    for k, d in enumerate(dets): # k = 얼굴 인식 갯수, d = 좌표(?)
        fX = d.left()
        fY = d.top()
        fW = d.right()
        fH = d.bottom()
        print(fX, fY, fW, fH)

        if (a, b, c, d, xf, yf, wf, hf) is not None:
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
        # --grabcut

        roi = gray[fY:fH, fX:fW] #얼굴 인식된 좌표
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # Emotion predict
        #감정 인식하기
        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]

        # Assign labeling
        cv2.putText(frame, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frame, (fX, fY), (fW, fH), (0, 0, 255), 2)


    # text는 배열(?)
    # Label printing
    #사람이 여려명일 때는 마지막으로 감정을 인식한 사람의 감정 상태만 프린트함
    maxnum = 1
    for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):

        text = "{}: {:.2f}%".format(emotion, prob * 100)
        if maxnum < prob * 100:
            maxnum = int(prob * 100)
        w = int(prob * 300)
        cv2.rectangle(canvas, (7, (i * 35) + 5), (w, (i * 35) + 35), (0, 0, 255), -1)
        cv2.putText(canvas, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

print(maxnum)
x = random.randint(0, 1)

# 이미지 지정
background = cv2.imread('C:/Users/user/PycharmProjects/OpenCV/Background/sky1.jpg')

output_height, output_width, _ = output.shape
print(output_height, output_width)
logo = output

background_height, background_width, _ = background.shape
logo_height, logo_width, _ = logo.shape

if background_height > 2 * logo_height:
    ratio = background_height * 0.5 / logo_height
    print(ratio)

    logo2 = cv2.resize(output, dsize=(0, 0), fx = ratio, fy = ratio, interpolation=cv2.INTER_LINEAR)
    logo2_height, logo2_width, _ = logo2.shape

    gray_logo = cv2.cvtColor(logo2, cv2.COLOR_BGR2GRAY)
    _, mask_inv = cv2.threshold(gray_logo, 10, 255, cv2.THRESH_BINARY_INV)

    x = (background_height - logo2_height)
    y = (background_width - logo2_width) // 2

    roi = background[x: x+logo2_height, y: y+logo2_width]

    cv2.imshow("ROI", roi)
    cv2.waitKey()

    roi_logo = cv2.add(logo2, roi, mask=mask_inv)
    cv2.imshow("roi_logo", roi_logo)
    cv2.waitKey()

    result = cv2.add(roi_logo, logo2)
    cv2.imshow("result", result)
    cv2.waitKey()

else:
    gray_logo = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
    _, mask_inv = cv2.threshold(gray_logo, 10, 255, cv2.THRESH_BINARY_INV)

    cv2.imshow("mask_inv", mask_inv)
    cv2.waitKey()

    background_height, background_width, _ = background.shape
    logo_height, logo_width, _ = logo.shape

    x = (background_height - logo_height)
    y = (background_width - logo_width) // 2

    roi = background[x: x + logo_height, y: y + logo_width]

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

# Open two windows
## Display image ("Emotion Recognition")
## Display probabilities of emotion
cv2.imshow('output', output)
if background_height > 2 * logo_height:
    cv2.imshow('output2', logo2)
cv2.imshow('Emotion Recognition', frame)
cv2.imshow("Probabilities", canvas)
cv2.waitKey(0)

# Clear program and close windows
cv2.destroyAllWindows()