import numpy as np
import cv2

cap = cv2.VideoCapture("vlog.mp4")
face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_alt2.xml")
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for(x, y, w, h) in faces:
        color_roi = frame[y:y+h, x:x+w]

        color = (255,0,0)
        cv2.rectangle(frame,(x,y),(x+w,y+h),color=color)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # q for exit()
        break
    cv2.imshow("frame",frame)


# when everything is done , release the capture
cap.release()
cv2.destroyAllWindows()
