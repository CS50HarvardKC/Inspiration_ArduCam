import cv2
from ultralytics import YOLO

def predict(frame ,model):
    detections = model(frame)

    # DEBUG
    for object in detections:
        print(type(object))
        print(object)



    
