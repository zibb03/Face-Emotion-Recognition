import numpy as np
import cv2

cap = cv2.VideoCapture('C:/Users/user/PycharmProjects/OpenCV/doc/chroma key.mp4')

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret2, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imshow('thr1',thr)

    # Morphology의 opening, closing을 통해서 노이즈나 Hole제거
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=2)

    # dilate를 통해서 확실한 Backgroud
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # distance transform을 적용하면 중심으로 부터 Skeleton Image를 얻을 수 있음.
    # 그 결과에 thresh를 이용하여 확실한 FG를 파악
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Background에서 Foregrand를 제외한 영역을 Unknow영역으로 파악
    unknown = cv2.subtract(sure_bg, sure_fg)

    # FG에 Labelling작업
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # watershed를 적용하고 경계 영역에 색지정
    markers = cv2.watershed(frame, markers)
    frame[markers == -1] = [0, 255, 0]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    cv2.imshow("h", h)
    # cv2.imshow("s", s)
    # cv2.imshow("v", v)'''

    cv2.imshow("gray", gray)
    cv2.imshow("sure_bg", sure_bg)
    # cv2.imshow("sure_fg", sure_fg)
    cv2.imshow("dist_transform", dist_transform)
    # cv2.imshow("unknown", unknown)
    cv2.imshow("frame2", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()