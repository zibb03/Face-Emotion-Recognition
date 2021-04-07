import cv2
from matplotlib import pyplot as plt
img2 = cv2.imread('C:/Users/user/PycharmProjects/COCO/person grayimage.jpg', 0)
img2_c = cv2.imread('C:/Users/user/PycharmProjects/COCO/person grayimage.jpg')

'''image = cv2.imread('C:/Users/user/PycharmProjects/COCO/person two.jpg', 0)
img2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
img2_c = cv2.imread('C:/Users/user/PycharmProjects/COCO/person two.jpg')
'''

thr, img2_mask = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
img2_mask = 255 - img2_mask

contours2, _ = cv2.findContours(img2_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img2_c, contours2, -1, (255, 0, 0), 10)

plt.imshow('Source', img2_mask)
plt.imshow('Contours', img2_c)
