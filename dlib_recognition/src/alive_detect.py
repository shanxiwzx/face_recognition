# -*- coding: utf-8 -*-
"""
活体检测
"""
import cv2
import dlib
import os
import time
import numpy as np
from imutils import face_utils
from scipy.spatial import distance as dist

start0 = time.time()
detector = dlib.get_frontal_face_detector()                     #
predictor = dlib.shape_predictor(
    "/home/wzx/PycharmProjects/AI/face_recognition/dlib_recognition/weights/shape_predictor_68_face_landmarks.dat"
)
end0 = time.time()
print("模型加载时间{}".format(end0-start0))

# 眼睛长宽比
# 闪烁阈值
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 3

def markland_detect(img):
    """
    @path: 输入路径
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)              #缩减计算时间
    dets = detector(gray, 1)                                  #返回面部坐标,多张脸的时候返回多个数组
    #dets, scores, idx = detector.run(img, 1)                 #scores分之越大,自信度越高
    print("检测到人脸数量{}个".format(len(dets)))   #打印人脸数目
    for i, d in enumerate(dets):
        shape = predictor(img, d)                             #检测人脸关键点
        shape = face_utils.shape_to_np(shape)
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]   #左眼关键点索引
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]  #右眼关键点索引
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        left_ratio = xy_ratio(leftEye)
        right_ratio = xy_ratio(rightEye)
        avg_ratio = (left_ratio+right_ratio)/2               #左右眼平均横纵比
        #con_loc = np.concatenate((leftEye, rightEye), axis=0)
        #print("比率:", avg_ratio)
        pts = np.array(leftEye, np.int32)
        pts1 = np.array(rightEye, np.int32)
        pts = pts.reshape((-1, 1, 2))
        pts1 = pts1.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, (0, 255, 255))
        cv2.polylines(img, [pts1], True, (0, 255, 255))
        # for dot in shape[42:48]:
        #     cv2.circle(img, (dot[0], dot[1]), 2, (255, 255, 0), 2)
    return avg_ratio


def xy_ratio(eye):
    """
    输入左右眼的坐标
    返回纵横比率
    """
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])      #左右两个角点的坐标
    ratio = (A+B)/(2*C)
    return ratio


def video_analysis(video):
    """
    解析视频为每一帧
    """
    print("[INFO] starting video stream thread...")
    cap = cv2.VideoCapture(video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    videoWriter = cv2.VideoWriter('video1.avi', cv2.VideoWriter_fourcc('I','4','2','0'), fps, size)

    COUNTER = 0  # 连续帧计数
    TOTAL = 0  # 眨眼数
    while True:
        ret, frame = cap.read()
        if ret:
            try:
                avg_ratio = markland_detect(frame)
                if avg_ratio and avg_ratio < EYE_AR_THRESH:
                    COUNTER += 1
                elif avg_ratio and COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1
                    COUNTER = 0
                else:
                    pass
                #print("眨眼{}次".format(TOTAL))
                cv2.putText(frame, "Blinks:{0}".format(TOTAL), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                            2)
                cv2.putText(frame, "EAR:{:.2f}".format(avg_ratio), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            except Exception as e:
                print("error:", e)
            # cv2.imshow('e', frame)
            # cv2.waitKey(0)
            videoWriter.write(frame)
        else:
            break
    cap.release()
    videoWriter.release()


if __name__ == "__main__":
    dir_path = r"/home/wzx/PycharmProjects/AI/face_recognition/dlib_recognition/sample"
    f = "dai.mp4"
    video_analysis(os.path.join(dir_path, f))
    #markland_detect(os.path.join(dir_path, f))
    print("Processing file: {}".format(f))

