import cv2
import numpy as np
import pickle

# configurations
class config:
    prototxt = "deploy.prototxt.txt"                             # this is the path of your deploy config file. eg: "deploy.prototxt.txt" for resnet model
    res_model = "res10_300x300_ssd_iter_140000.caffemodel"       # this is the path of your model file. eg: "res10_300x300_ssd_iter_140000.caffemodel" for resnet model
    video_path = "vlog2.mp4"                                     # this is the path where you video is stored
    confidence_rate = 0.3                                        # set a confidence rate to filter out weak predictions

if __name__ == "__main__":

    # load pre-trained recognizer model
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("model.yml")

    # get labels
    labels = {}
    with open("labels.pickle", "rb") as f:
        labels = pickle.load(f)
        labels = {k:v for v,k in labels.items()}

    # read the model and the source video/image
    net = cv2.dnn.readNetFromCaffe(config.prototxt, config.res_model)
    print("loading model ...")
    cap = cv2.VideoCapture(config.video_path)
    print("loading video/image ...")

    # loop over every frame
    image_num = 0
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = np.array(gray,"uint8")
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # filter out weak detections
            if confidence < config.confidence_rate:
                continue

            # compute the coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # predict
            roi = gray[startY:endY, startX:endX]              # region of interest
            id, conf = recognizer.predict(roi)

            text = "Unknown"
            res = False
            if conf > 80:
                text = labels[id]
                res = True

            # draw the bounding box of the face along with the predicted category
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 0, 255), 2)

            if res:
                cv2.imwrite("{}.jpg".format(image_num),frame)
                image_num += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):      # q for exit()
            break

        cv2.imshow("frame", frame)

    # when everything is done , release the capture
    cap.release()
    cv2.destroyAllWindows()