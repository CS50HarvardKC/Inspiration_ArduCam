"""
Buoy CV logic, tested on training videos.

Author: Keith Chen
"""
import cv2
import time
import numpy as np
import os
import preprocess

# Import the video 
# Read each frame
# Detect the buoy through color thresholding
# Return motion values

class CV:
    def __init__(self):
        self.shape = (640, 480)

    def detect_green_buoy(self, frame):
        """
        Uses HSV color space and masking to detect a red object. Returns bounding box coordinates and the visualized frame.
        """
        detected = False

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define lower and upper bounds for green color in HSV
        lower_green = np.array([70, 25, 55])  # Adjust values if needed
        upper_green = np.array([155, 255, 255])

        # Create a mask for green color
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw bounding boxes around detected green objects

        if contours:
            for contour in contours:
                if cv2.contourArea(contour) > 0:  # Adjust area threshold as needed
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                else:
                    x, y, w, h = (0, 0, 0, 0)

                return {"status": detected, "xmin" : (x), "xmax" : (x + w), "ymin" : (y), "ymax" : (y + h)}, frame
            
        else:
            return _, frame
            
        # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # lower_red_mask_1 = np.array([0, 120, 70])
        # upper_red_mask_1 = np.array([10, 255, 255])
        # lower_red_range_mask = cv2.inRange(hsv, lower_red_mask_1, upper_red_mask_1)

        # lower_red_mask_2 = np.array([170, 120, 70])
        # upper_red_mask_2 = np.array([180, 255, 255])
        # upper_red_range_mask = cv2.inRange(hsv, lower_red_mask_2, upper_red_mask_2)

        # mask = lower_red_range_mask + upper_red_range_mask

        # # Find contours
        # contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # if contours:
        #     largest_contour = max(contours, key = cv2.contourArea)

        #     if cv2.contourArea(largest_contour) > 0:
        #         x, y, w, h = cv2.boundingRect(largest_contour)
        #         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #         detected = True
        #         return {"status": detected, "xmin" : x, "xmax" : (x + w), "ymin" : (y), "ymax" : (y + h)}, frame
        
        # return {"status": detected, "xmin" : None, "xmax" : None, "ymin" : None, "ymax" : None}, frame        

 
if __name__ == "__main__":
    # "C:\Users\netwo\Downloads\03_h264.mp4"
    video_root_path = r"C:\Users\netwo\Downloads\\"
    # mission_name = "Buoy/"
    video_name = "02_h264.mp4"
    video_path = os.path.join(video_root_path, video_name)
    print(f"Video path: {video_path}")

    cv = CV()

    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found {video_path}")
    else:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Unable to open video file: {video_path}")
        else:
            alpha = 0.1
            current_Y_mean = 240
            reference_Y_mean = current_Y_mean
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[INFO] End of file.")
                    break
                reference_Y_mean = alpha * current_Y_mean + (1 - alpha) * reference_Y_mean if reference_Y_mean else current_Y_mean
                # print(reference_Y_mean)
                current_Y_mean = reference_Y_mean

                cropped_frame = preprocess.crop(frame)
                preprocessed_frame = preprocess.balance(frame, current_Y_mean)

                detection_info, viz_frame = cv.detect_green_buoy(preprocessed_frame)
                # detection_info, viz_frame = cv.detect_green_buoy(frame)
                print(detection_info)
                if viz_frame is not None:
                    cv2.imshow("frame", viz_frame)
                else:
                    print("[ERROR] Unable to display frame.")

                # For testing purposes.
                
                time.sleep(0.03)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break