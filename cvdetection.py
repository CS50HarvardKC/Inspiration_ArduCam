import cv2
import numpy as np

def preprocess_frame(frame):
    """ Convert to HSV and apply Gaussian Blur """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
    return hsv

def get_red_mask(hsv):
    """ Create a binary mask for detecting red objects in HSV """
    # Define red color range (two parts due to HSV wrap-around)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for both red ranges
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # Combine both masks
    mask = mask2

    # Apply morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def get_green_mask(hsv):
    """ Create a binary mask for detecting green objects in HSV """
    # Define green color range
    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])

    # Create mask for the green range
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Apply morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask

def detect_objects(frame, mask):
    """ Find contours and draw bounding boxes around detected red objects """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:  # Filter small objects
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box
            cv2.putText(frame, "Red Object", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (0, 255, 0), 2)

    return frame

def find_extreme(mask):
    """find left and right most object"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Initialize variables to store extreme objects
    leftmost_x = float('inf')
    rightmost_x = float('-inf')
    leftmost_contour = None
    leftmost_center = None
    rightmost_contour = None
    rightmost_center = None


    # Loop through contours to find the extreme objects
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Filter small objects
            x, y, w, h = cv2.boundingRect(contour)  # Get bounding box

            if x < leftmost_x:  # Update leftmost object
                leftmost_x = x
                leftmost_contour = contour
                leftmost_center = x + w//2

            if x + w > rightmost_x:  # Update rightmost object
                rightmost_x = x + w
                rightmost_contour = contour
                rightmost_center = x + w//2

    return leftmost_center,rightmost_center

def find_lowest_red(mask, normalize=False):
    """
    Finds the lowest red pixel in the given binary mask and normalizes the coordinates to a 0-1 range.

    :param mask: Binary mask with red pixels as white (255).
    :param normalize: Whether to normalize the coordinates to a 0-1 range.
    :return: (x, y) coordinates of the lowest red pixel, or None if not found.
    """
    height, width = mask.shape

    # Find all nonzero (white) pixel locations
    coordinates = np.column_stack(np.where(mask > 0))

    if coordinates.shape[0] == 0:
        return None  # No red pixels found

    # Get the pixel with the maximum y-coordinate (lowest in the image)
    lowest_pixel = tuple(coordinates[np.argmax(coordinates[:, 0])])  # (y, x) format

    # Normalize if required
    if normalize:
        # Normalize to 0-1 range
        normalized_x = np.clip(lowest_pixel[1] / width, 0, 1)
        normalized_y = np.clip(lowest_pixel[0] / height, 0, 1)
        lowest_pixel = (normalized_x, normalized_y)

    return lowest_pixel  # Return normalized (x, y) or absolute (x, y)

class cvCore:
    def __init__(self,port:str = "/dev/video0"):
        self.cap = cv2.VideoCapture(port)  #"/home/chaser/Downloads/02_h264.mp4"
        self.lock = threading.Lock


    def control_loop_test(self):
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            return

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            hsv = preprocess_frame(frame)
            red_mask = get_red_mask(hsv)
            green_mask = get_green_mask(hsv)

            red_result = detect_objects(frame, red_mask)
            green_result = detect_objects(frame, green_mask)

            Rleft,Rright = find_extreme(red_mask)
            Gleft,Gright = find_extreme(green_mask)

            # Show output
            cv2.imshow("Original", frame)
            cv2.imshow("RMask", red_mask)
            cv2.imshow("GMask", green_mask)

            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
    
    def control_loop(self,motor=None,dir=None,debug=False):
        # dir -> left or right
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            return

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            hsv = preprocess_frame(frame)
            mask = get_red_mask(hsv)

            result = detect_objects(frame, mask)

            leftmost,rightmost = find_extreme(mask)

            if dir=="left":
                if(leftmost<640*0.2):
                    print("DEBUG: Turn left")
                    if not debug:
                        motor.yaw(1,-0.5)
                else:
                    if not debug:
                        motor.surge(1)
            else:
                if(rightmost>640*0.8):
                    print("DEBUG: Turn right")
                    if not debug:
                        motor.yaw(1,0.5)
                else:
                    if not debug:
                        motor.surge(1)

            cv2.imshow("mask", mask)
            time.sleep(1/20)








def main():
    """ Main function for real-time red object detection """
    cap = cv2.VideoCapture("/home/chaser/Downloads/02_h264.mp4")  # Change index if needed for different cameras

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = preprocess_frame(frame)
        mask = get_red_mask(hsv)

        result = detect_objects(frame, mask)

        low_red = find_lowest_red(mask,normalize=True)
        if low_red is not None:
            print(f"x: {low_red[0]} | y: {low_red[1]}")

        # Show output
        cv2.imshow("Original", frame)
        cv2.imshow("Mask", mask)
        cv2.imshow("result", result)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import threading
    import time
    cam = cvCore()
    cam_thread = threading.Thread(target=cam.control_loop_test,daemon=True)
    cam_thread.start()

    for i in range(120):
        print(f"time: {i}")
        time.sleep(1)
