import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import dlib
#dlib를 상용 하는 이유는 XML파일을 이용하는 것 보다 dlib를 사용하여 인식 하는 것이 더 인식이 잘되는 듯 함
#dlib를 설치하기 위해서는 Anaconda에서 pip install cmake -> pip install dlib 설치


# dlib얼굴 인식 모델, 감정인식 모델 불러오기
#face_detection = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml') #사용 안함
detector = dlib.get_frontal_face_detector() #얼굴 인식을 위한 dlib 기본 안면인식 모델
emotion_classifier = load_model('./emotion/emotion_model.hdf5', compile=False)
EMOTIONS = ["Angry", "Disgusting", "Fearful", "Happy", "Sad", "Surpring", "Neutral"]


# 이미지 불러오기
frame = cv2.imread('./SampleImage/peoples.jpg')
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
canvas = np.zeros((250, 300, 3), dtype="uint8")
dets = detector(frame, 1) #얼굴인식 완료

# Perform emotion recognition only when face is detected
if len(dets) > 0:
    # Resize the image to 48x48 for neural network
    for k, d in enumerate(dets): # k = 얼굴 인식 갯수, d = 좌표(?)
        fX = d.left()
        fY = d.top()
        fW = d.right()
        fH = d.bottom()
        print(fX, fY, fW, fH)

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

    # Label printing
    #사람이 여려명일 때는 마지막으로 감정을 인식한 사람의 감정 상태만 프린트함
    for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
        text = "{}: {:.2f}%".format(emotion, prob * 100)
        w = int(prob * 300)
        cv2.rectangle(canvas, (7, (i * 35) + 5), (w, (i * 35) + 35), (0, 0, 255), -1)
        cv2.putText(canvas, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

# Open two windows
## Display image ("Emotion Recognition")
## Display probabilities of emotion
cv2.imshow('Emotion Recognition', frame)
cv2.imshow("Probabilities", canvas)
cv2.waitKey(0)



# Clear program and close windows
cv2.destroyAllWindows() 