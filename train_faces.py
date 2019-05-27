import os
from PIL import Image
import numpy as np
from detect_faces import config
import cv2
import pickle


dir_name = "faces"                                  # this is the dir name where you store the folders of iamges
BaseDir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(BaseDir, dir_name)

# detector configuration
prototxt = config.prototxt                          # this is the path of your deploy config file. eg: "deploy.prototxt.txt" for resnet model
res_model = config.res_model                        # this is the path of your model file. eg: "res10_300x300_ssd_iter_140000.caffemodel" for resnet model
confidence_rate = config.confidence_rate
net = cv2.dnn.readNetFromCaffe(prototxt, res_model)

# define the recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
label_id = 0
train = []
label = []
label_dic = {}

# loop through all images for training and store the image arrays and labels and label-name dictionary
for root, dirs, files in os.walk(image_path, topdown=False):
    for file in files:
        name = os.path.basename(root)
        path = os.path.join(root,file)
        if not name in label_dic:
            label_dic[name] = label_id
            label_id += 1

        id = label_dic[name]
        image = Image.open(path)
        gray = image.convert("L")
        gray = np.array(gray,"uint8")
        image = np.array(image,"uint8")

        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # filter out weak detections
            if confidence < confidence_rate:
                continue

            # compute the coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

            train.append(gray[int(box[1]):int(box[3]),int(box[0]):int(box[2])])
            label.append(id)

# train and save the model

recognizer.train(train,np.array(label))
recognizer.save("model.yml")

# save name-lable dictionary
with open("labels.pickle","wb") as f:
    pickle.dump(label_dic,f)



