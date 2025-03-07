import cv2
import preprocess

def main():
    # Open the Arducam video stream (adjust device index if needed)
    cap = cv2.VideoCapture("/dev/video0")  # Use /dev/videoX if necessary
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        cropped_frame = preprocess.crop(frame)
        preprocessed_frame = preprocess.balance(frame)
        
        cv2.imshow("Arducam Video", frame)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()