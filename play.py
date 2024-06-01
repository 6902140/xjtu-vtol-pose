#這是西安交通大學 智能飛行器社團 2024 年 校園開放日視覺方向對外開放項目
#是一個基於yolo-pose開發的體感小遊戲
#鳴謝：developer：計試2101 肖追日、程煜博

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QFileDialog, QPushButton
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer
import cv2
from ultralytics import YOLO
from pyorbbecsdk import Pipeline, Config, OBSensorType, FrameSet, OBFormat, OBPropertyID
import numpy as np
import threading
import random
from PyQt5 import QtGui

SPEED=20

class VideoPlayer(QWidget):
    def __init__(self):
        super().__init__()
        
        self.model = YOLO("/home/moon/Downloads/yolov8n-pose.pt")
        self.score=0
        self.points=[]
        while len(self.points) < 5:    
            self.points.append([random.randint(40, 2000), random.randint(0,500)])
        self.initUI()

    def initUI(self):
        
        self.currentFrame = None
        self.setWindowTitle("Video Analyzer")
        self.setGeometry(100, 100, 800, 800)
        
        
        # 創建垂直佈局
        main_layout = QVBoxLayout()
        
        # 顯示得分
        self.score_label = QLabel(f"Score: {self.score}")
        main_layout.addWidget(self.score_label)
        font = QtGui.QFont()
        font.setFamily("Arial") # 设置字体
        font.setPointSize(20)
        font.setBold(True) # 设置为粗体
        self.score_label.setFont(font)

        # 上方顯示原始視頻
        # self.original_video_label = QLabel()
        # self.original_video_label.setAlignment(Qt.AlignCenter)
        # self.original_video_label.setFixedSize(640, 480)
        # main_layout.addWidget(self.original_video_label)

        # 下方顯示識別後的視頻
        self.processed_video_label = QLabel()
        self.processed_video_label.setAlignment(Qt.AlignCenter)
        self.processed_video_label.setScaledContents(True)
        self.processed_video_label.setFixedSize(1200, 900)
        main_layout.addWidget(self.processed_video_label)

        self.setLayout(main_layout)

        # 開啟文件對話框
        open_file_button = QPushButton("Open Video")
        open_file_button.clicked.connect(self.open_video)
        main_layout.addWidget(open_file_button)
        close_button = QPushButton("Restart")
        close_button.clicked.connect(self.close_video)
        main_layout.addWidget(close_button)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_video)
        self.cap = None
        self.processed_cap = None

        # orbbec 攝像

        self.pipeline = Pipeline()
        self.device = self.pipeline.get_device()
        self.config = Config()

        print("正在初始化相机")
        self.device.set_bool_property(OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, False)
       
        self.device.set_int_property(OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT, 10)
       
        try:
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            color_profile = profile_list.get_video_stream_profile(2048, 0, OBFormat.RGB, 15)
            self.config.enable_stream(color_profile)
            self.pipeline.start(self.config)
        except Exception as e:
            print(e)
            exit()
        self.camera_running = True
        self.camera_thread = threading.Thread(target=self.camera_thread_run)
        self.camera_thread.start()

    def camera_thread_run(self):
        while True:
            if not self.camera_running:
                continue
            frames: FrameSet = self.pipeline.wait_for_frames(100)
            if frames is None:
                continue
            color_frame = frames.get_color_frame()
            if color_frame is None:
                continue
            width = color_frame.get_width()
            height = color_frame.get_height()
            data = np.asanyarray(color_frame.get_data())
            color_image = np.resize(data, (height, width, 3))
            # color_image = color_image[:, :, ::-1]
            if color_image is None:
                print("failed to convert frame to image")
                continue
            self.currentFrame = color_image

    def open_video(self):
        self.timer.start(30)  # 30 FPS
    
    def close_video(self):
        self.score=0

    def update_video(self):
        frame = self.currentFrame
        if frame is not None:
            for i in range(len(self.points)):
                self.points[i][1] += SPEED
            self.points=[point for point in self.points if point[1]<1400]
            if len(self.points) < 5:
                self.points.append([random.randint(40, 2000), 0])
            image_data = np.array(frame)
            # 转换为 BGR 格式
            bgr_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
            bgr_data=cv2.flip(bgr_data, 1)
            # 创建 QImage 对象
            image = QImage(bgr_data.data, bgr_data.shape[1], bgr_data.shape[0], bgr_data.shape[1] * 3, QImage.Format_BGR888)
            pixmap = QPixmap.fromImage(image)
            # self.original_video_label.setPixmap(pixmap.scaled(640, 480))
            results = self.model.predict(frame, save=False, imgsz=640, conf=0.5)
            processed_frame = results[0].plot(kpt_radius=20)
            for pose in results[0].keypoints.xy[0]:
                self.points=[point for point in self.points if (point[0]-pose[0])**2+(point[1]-pose[1])**2>6000]
            self.score += 5 - len(self.points)
            for point in self.points:
                cv2.circle(processed_frame, (int(point[0]), int(point[1])), 50, (255, 0, 0), thickness=cv2.FILLED)
            # 将 RGB 数据转换为 BGR
            bgr_data = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
            bgr_data=cv2.flip(bgr_data, 1)

            # 创建 QImage 对象
            processed_image = QImage(bgr_data.data, bgr_data.shape[1], bgr_data.shape[0], bgr_data.shape[1] * 3, QImage.Format_BGR888)

            # 创建 QPixmap 对象
            processed_pixmap = QPixmap.fromImage(processed_image)
            self.processed_video_label.setPixmap(processed_pixmap.scaled(640, 480))
            self.score_label.setText(f"Score: {self.score}")
            # 暫停 33 毫秒以達到 30 FPS 的效果
            self.timer.start(33)


    def clear_score(self):
        self.score = 0
        self.score_label.setText(f"Score: {self.score}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec_())
