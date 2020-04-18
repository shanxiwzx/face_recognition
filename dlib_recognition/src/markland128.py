# -*- coding: utf-8 -*-
"""
检测人脸关键点,提取人脸72个关键点坐标,进行简单化妆
batch_face_locations:可以批次进行多张人脸的检测
"""
import face_recognition as fr
import os
import cv2
import time
import numpy as np
from PIL import ImageDraw, Image

dir_path = r"/home/wzx/PycharmProjects/AI/face_recognition/dlib_recognition/sample"
f = "women_smile_2020_1.jpg"
#img = fr.load_image_file(os.path.join(dir_path, f))
img = cv2.imread(os.path.join(dir_path, f))
#img1 = Image.open(os.path.join(dir_path, f))
face_location = fr.face_locations(img)               #获取人脸位置
# start = time.time()
# face_marks = fr.face_landmarks(img)
# end = time.time()
#print("关键点检测耗时{}".format(end-start))

#print("检测到人脸关键点坐标{}个".format(n))
# cv2.imshow('w', img)
# cv2.waitKey(0)
for face_loc in face_location:
    top, right, bottom, left = face_loc
    face_img = img[top-50:bottom+50, left-50:right+50]
    #cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 3)

# cv2.imshow('w', face_img)
# cv2.waitKey(0)
start = time.time()
face_marks = fr.face_landmarks(img)
#print(face_marks)
end = time.time()
print("关键点检测耗时{}".format(end-start))
for mark in face_marks:     #遍历每一个人脸
    for k in mark:          #遍历每一个区域
        for d in mark[k]:
            cv2.circle(img, (d[0], d[1]), 2, (255, 255, 0), 2)
cv2.imshow('w', img)
cv2.waitKey(0)