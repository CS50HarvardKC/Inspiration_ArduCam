import cv2
from ultralytics import YOLO


class Camera:
    def __init__(self):
        self.cam        = None
        self.fps        = None
        self.width      = None
        self.height     = None
        self.resolution = None

        self.detections = None

    def start(self):
        pass

    def stop(self):
        pass
        