import cv2
import cv2 as cv

img = cv2.imread("C:/Users/user/PycharmProjects/OpenCV/capture.jpg", cv2.IMREAD_COLOR) #흑백이미지 https://076923.github.io/posts/Python-opencv-11/

img_g = cv2.bitwise_not(img) #반전이미지 만드는 과정

cv2.imshow("img_g", img_g)
cv2.waitKey(0)
cv2.destroyAllWindows()

img_color = cv.imread('C:/Users/user/PycharmProjects/OpenCV/persontwo.jpg')
img_gray = cv.cvtColor(img_g, cv.COLOR_BGR2GRAY)
cv2.imshow("img_gray", img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()


ret, img_binary = cv.threshold(img_gray, 127, 255, 0) #https://webnautes.tistory.com/1270
contours, hierarchy = cv.findContours(img_binary, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

for cnt in contours: #contours 파일이 원하는 대상이 흰색일 때 잘 되어 반전이미지 처리함
    cv.drawContours(img_color, [cnt], 0, (255, 0, 0), 3)  # blue

cv.imshow("result", img_color)

cv.waitKey(0)


for cnt in contours:
    area = cv.contourArea(cnt)
    print(area)

cv.imshow("result", img_color)

cv.waitKey(0)