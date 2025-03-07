import cv2
from ultralytics import YOLO

def predict(frame ,model): # pass in frame and yolo model
    detections = model(frame)

    # DEBUG
    for object in detections:
        print(type(object))
        print(object)

    return detections



    
