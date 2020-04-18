# -*- coding: utf-8 -*-
"""
检测人脸,并返回人脸区域
"""
import cv2
import dlib
import os
import time


detector = dlib.get_frontal_face_detector()                     #
predictor = dlib.shape_predictor(
    "/home/wzx/PycharmProjects/AI/face_recognition/dlib_recognition/weights/shape_predictor_68_face_landmarks.dat"
)

dir_path = r"/home/wzx/PycharmProjects/AI/face_recognition/dlib_recognition/sample/videos"
f = "18.jpg"
print("Processing file: {}".format(f))
start = time.time()
img = cv2.imread(os.path.join(dir_path, f))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)              #缩减计算时间
dets = detector(gray, 1)                                  #返回面部坐标,多张脸的时候返回多个数组
#dets, scores, idx = detector.run(img, 1)                 #scores分之越大,自信度越高
#face_descriptor = face_rec_model.compute_face_descriptor(img, shape)   # 计算人脸的128维的向量
print(dets)
print("Number of faces detected: {}".format(len(dets)))   #打印人脸数目
for i, d in enumerate(dets):
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        i, d.left(), d.top(), d.right(), d.bottom()))
    img1 = img[d.top():d.bottom(), d.left():d.right()]
    end = time.time()
    print("计算时间：", end-start)
    cv2.imshow('wzx', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#dlib.hit_enter_to_continue()




