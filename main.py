import cv2
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt",required=True,help="path to caffe prototxt file")
ap.add_argument("-m","--model",required=True,help="path to caffe file")
ap.add_argument("-c","--confidence",type=float,default=0.4,help="min probability to filter out weak predictions")
args = vars(ap.parse_args())

net = cv2.dnn.readNetFromCaffe(args["prototxt"],args["model"])

cap = cv2.VideoCapture("vlog.mp4")

while True:
    ret, frame = cap.read()
    # frame = imutils.resize(frame, width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (224, 224), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence < args["confidence"]:
            continue

        # compute the (x, y)-coordinates of the bounding box for the object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # draw the bounding box of the face along with the associated probability
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY),
                      (0, 0, 255), 2)
        cv2.putText(frame, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # q for exit()
            break
        cv2.imshow("frame", frame)

# when everything is done , release the capture
cap.release()
cv2.destroyAllWindows()