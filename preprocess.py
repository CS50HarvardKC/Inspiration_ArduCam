"""preprocessing: crop + color balance(optional)"""
import cv2
import numpy as np

def crop(frame):
    w,h = (frame.shape[0], frame.shape[1])
    size = 640
    cropped_frame = frame[(h-640):h,(1920-640)//2:((1920-640)//2 + 640)]
    return cropped_frame

def balance(frame, reference_Y_mean = None):
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    current_Y_mean = ycrcb[:, :, 0].mean()
    print(reference_Y_mean)

    if reference_Y_mean is not None and current_Y_mean > 0:
        gamma = reference_Y_mean / current_Y_mean
        invGamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** invGamma * 255 for i in range(256)]).astype("uint8")
        ycrcb[:, :, 0] = cv2.LUT(ycrcb[:, :, 0], table)

    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

