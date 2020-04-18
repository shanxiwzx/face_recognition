# -*- coding: utf-8 -*-
"""
检测人脸关键点,提取人脸68个关键点坐标
"""
import cv2
import dlib
import os
import time

start0 = time.time()
detector = dlib.get_frontal_face_detector()                     #
predictor = dlib.shape_predictor(
    "/home/wzx/PycharmProjects/AI/face_recognition/dlib_recognition/weights/shape_predictor_68_face_landmarks.dat"
)
convert_128 = dlib.face_recognition_model_v1("/home/wzx/PycharmProjects/AI/face_recognition/dlib_recognition/"
                                         "weights/dlib_face_recognition_resnet_model_v1.dat")
end0 = time.time()
print("模型加载时间{}".format(end0-start0))

dir_path = r"/home/wzx/PycharmProjects/AI/face_recognition/dlib_recognition/sample"
f = "women_smile_2020_34.jpg"
print("Processing file: {}".format(f))
#start = time.time()
img = cv2.imread(os.path.join(dir_path, f))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)              #缩减计算时间
start1 = time.time()
dets = detector(gray, 1)                                  #返回面部坐标,多张脸的时候返回多个数组
end1 = time.time()
print("人脸区域检测用时{}".format(end1-start1))
#dets, scores, idx = detector.run(img, 1)                 #scores分之越大,自信度越高
print("检测到人脸数量{}个".format(len(dets)))   #打印人脸数目
for i, d in enumerate(dets):
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        i, d.left(), d.top(), d.right(), d.bottom()))
    # img1 = img[d.top()-10:d.bottom()+20, d.left()-10:d.right()+10]
    # h, w, _ = img1.shape
    # t = dlib.rectangle(0, 0, w, h)                         #制作成dlib.rectangle人脸坐标
    # start2 = time.time()
    # shape = predictor(img1, t)                             #只在人脸区域进行检测
    # end2 = time.time()
    start2 = time.time()
    shape = predictor(img, d)                             #检测人脸关键点
    #face_vector = convert_128.compute_face_descriptor(img, shape)
    #print(len(face_vector))
    end2 = time.time()
    print("人脸关键点检测用时{}".format(end2-start2))
    print("检测到人脸关键点{}个".format(len(shape.parts())))
    print(shape.parts())
    for dot in shape.parts()[42:48]:
        cv2.circle(img, (dot.x, dot.y), 2, (255, 255, 0), 2)
    cv2.imshow('wzx', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
