import numpy as np
import cv2
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import dlib
#dlib를 상용 하는 이유는 XML파일을 이용하는 것 보다 dlib를 사용하여 인식 하는 것이 더 인식이 잘되는 듯 함
#dlib를 설치하기 위해서는 Anaconda에서 pip install cmake -> pip install dlib 설치


# dlib얼굴 인식 모델, 감정인식 모델 불러오기
face_detection = cv2.CascadeClassifier('C:/Users/user/PycharmProjects/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
#detector = dlib.get_frontal_face_detector() #얼굴 인식을 위한 dlib 기본 안면인식 모델
emotion_classifier = load_model('C:/Users/user/PycharmProjects/OpenCV/emotion_model.hdf5', compile=False)
EMOTIONS = ["Angry", "Disgusting", "Fearful", "Happy", "Sad", "Surpring", "Neutral"]


cap = cv2.VideoCapture('C:/Users/user/PycharmProjects/OpenCV/doc/watershed_TestVideo/fail_1.mp4')

x, y = int(cap.get(3)), int(cap.get(4))
bgd = cv2.imread('C:/Users/user/PycharmProjects/OpenCV/Background/sky1.jpg')
bgd = cv2.resize(bgd, (x, y))
while True:
    ret, frame = cap.read()
    ret, src = cap.read()
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray,
                                            scaleFactor=1.1,
                                            minNeighbors=5,
                                            minSize=(30, 30))
    canvas = np.zeros((250, 300, 3), dtype="uint8")


    if len(faces) > 0:
        # For the largest image
        face = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = face
        # Resize the image to 48x48 for neural network
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]

        # Assign labeling
        cv2.putText(src, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(src, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)

        # Label printing
        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
            text = "{}: {:.2f}%".format(emotion, prob * 100)
            w = int(prob * 300)
            cv2.rectangle(canvas, (7, (i * 35) + 5), (w, (i * 35) + 35), (0, 0, 255), -1)
            cv2.putText(canvas, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)


    ret2, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

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
    a[dilation != 255] = (0, 0, 0)


    mask = cv2.inRange(a, upper_white, upper_white)
    a[mask == 255] = (0,0,0)

    res = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.imshow('res', res)

    a = cv2.add(a, res)
    cv2.imshow('a', a)

    b = cv2.add(a, bgd, mask=binary)
    b[mask == 255] = (0,0,0)
    cv2.imshow('b', b)

    result = cv2.add(a,b)
    cv2.imshow('result',result)
    cv2.imshow('Emotion Recognition', src)
    cv2.imshow("Probabilities", canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()