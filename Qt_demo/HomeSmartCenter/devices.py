import cv2
from PyQt5.QtWidgets import QMessageBox

class Camera:

    def __init__(self, camera):
        self.camera = camera
        self.cap = None

    def openCamera(self):
        self.vc = cv2.VideoCapture(0)
        # vc.set(5, 30)  #set FPS
        self.vc.set(3, 640)  # set width
        self.vc.set(4, 480)  # set height

        if not self.vc.isOpened():
            print('failure')
            msgBox = QMessageBox()
            msgBox.setText("Failed to open camera.")
            msgBox.exec_()
            return

    # https://stackoverflow.com/questions/41103148/capture-webcam-video-using-pyqt
    def initialize(self):
        self.cap = cv2.VideoCapture(self.camera)