import cv2
import numpy as np
img=cv2.imread("image.jpg")
img=cv2.resize(img,(400,400))
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray=cv2.medianBlur(gray,5)
edge= cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,9,9)
color=cv2.bilateralFilter(img,9,250,250)
cartoon=cv2.bitwise_and(color,color,mask=edge)
cv2.imshow("original",img)
cv2.imshow("edges",edge)
cv2.imshow("cartoon",cartoon)
cv2.waitKey(0)
cv2.destroyAllWindows()