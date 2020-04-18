"""
面部表情识别
"""
import cv2
from model import FacialExpressionModel
import dlib
import numpy as np
import argparse
parser = argparse.ArgumentParser(description="person image")
parser.add_argument('-p', '--path', default='/home/wzx/PycharmProjects/AI/face_recognition/dlib_recognition/sample/four_man.jpeg')
args = parser.parse_args()


detector = dlib.get_frontal_face_detector()
model = FacialExpressionModel("model.json", "model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX

img = cv2.imread(args.path)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dets = detector(img_gray, 2)

for d in dets:
    face_area = img_gray[d.top():d.bottom(), d.left():d.right()]
    roi = cv2.resize(face_area, (48, 48))
    pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])   #得到标签
    #print(pred)
    cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 0), 3)
    #x_put = d.left() + (d.right()-d.left())/2
    cv2.putText(img, pred, (d.left(), d.top()), font, 1, (255, 255, 255), 2)
cv2.imshow('t', img)
cv2.waitKey(0)

