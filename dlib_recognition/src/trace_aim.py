# -*- coding: utf-8 -*-
"""
目标跟踪
"""
import dlib
from imageio import imread
import glob
import cv2
import os

dir_path = r"/home/wzx/PycharmProjects/AI/face_recognition/dlib_recognition/sample"
f = "dai.mp4"
path_video = os.path.join(dir_path, f)
def video_split(path_video):
    """
    讲视频解析成每一帧图片并保存
    """
    cap = cv2.VideoCapture(path_video)
    n = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        n += 1
        path = os.path.join(dir_path, 'videos')
        path1 = os.path.join(path, str(n)+".jpg")
        cv2.imwrite(path1, frame)


if __name__ == "__main__":
    tracker = dlib.correlation_tracker()
    cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
    paths = sorted(glob.glob('../sample/videos/*.jpg'))  #glob.glob获取指定路径下的所有jpg文件，将视频先解析为.jpg

    for i, path in enumerate(paths):
        #print(path)
        img = imread(path)
        # 第一帧，指定一个区域
        if i == 0:
            tracker.start_track(img, dlib.rectangle(225, 239, 354, 368))
        # 后续帧，自动追踪
        else:
            tracker.update(img)
        box_predict = tracker.get_position()
        cv2.rectangle(img, (int(box_predict.left()), int(box_predict.top())),
                      (int(box_predict.right()), int(box_predict.bottom())), (0, 255, 255), 1)  # 用矩形框标注出来

        cv2.imshow("image", img)
        # 如果按下ESC键，就退出
        if cv2.waitKey(10) == 27:
            break
    cv2.destroyAllWindows()