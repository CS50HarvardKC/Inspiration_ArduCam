import cv2
from ultralytics import YOLO
from preprocess import balance, crop
from detect import predict

class Camera:
    def __init__(self, MLPath:str):
        self.cam        = None
        self.fps        = None
        self.width      = None
        self.height     = None

        self.model_path = MLPath
        self.device     = "cpu"
        self.model      = None
        self.detections = None

    def start(self):
        # set camera properties
        self.cam = cv2.VideoCapture("/dev/video0")
        self.fps = self.cam.get(cv2.CAP_PROP_FPS)
        self.width = self.cam.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # set yolo properties
        self.model = YOLO(self.model_path).to(self.device)

    def stop(self):
        self.cam.release()

    def main(self):
        """This will be our target function for thread"""
        last_deteciton_time = 0

        while(True):
            # read fream from camera
            ret, frame = self.cam.read()
            if not ret:
                print("ERROR: Frame Not Found")

            # preprocess
            # cropped_frame = crop(frame)

            # predict
            detections = self.model(frame)

            for object in detections:
                object.plot()
                print(object)

            cv2.imshow("cam", frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
    


if __name__ == "__main__":
    cam = Camera(MLPath="competition.pt")
    cam.start()

    cam.main()

        