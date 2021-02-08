import cv2
import numpy as np

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade=cv2.CascadeClassifier("haarcascade_eye.xml")
smile_cascade=cv2.CascadeClassifier("haarcascade_smile.xml")

cap=cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

while True:
    ret,img=cap.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces=face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),4)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        eyes= eye_cascade.detectMultiScale(roi_gray,1.5,5)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),3)

        smile=smile_cascade.detectMultiScale(roi_gray,1.5,25)
        for (sx,sy,sw,sh) in smile:
            cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(0,255,0),3)

        cv2.imshow('video',img)

    if cv2.waitKey(10) & 0xff==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


