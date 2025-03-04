#!/usr/bin/env python3
import cv2
import depthai as dai

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and output
camRgb = pipeline.create(dai.node.ColorCamera)
videoEnc = pipeline.create(dai.node.VideoEncoder)
xout = pipeline.create(dai.node.XLinkOut)

xout.setStreamName('h265')

# Properties
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1200_P)
camRgb.setFps(30)  # Set FPS to 30
videoEnc.setDefaultProfilePreset(30, dai.VideoEncoderProperties.Profile.H265_MAIN)

# Linking
camRgb.video.link(videoEnc.input)
videoEnc.bitstream.link(xout.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queue will be used to get the encoded data from the output defined above
    q = device.getOutputQueue(name="h265", maxSize=30, blocking=True)

    # The .h265 file is a raw stream file (not playable yet)
    with open('video.h265', 'wb') as videoFile:
        print("Press Ctrl+C to stop encoding...")
        try:
            while True:
                h265Packet = q.get()  # Blocking call, will wait until new data arrives
                cvFrame = h265Packet.getCvFrame()  # This gets the raw frame from the camera
                cv2.imshow("camera", cvFrame)  # Display the decoded frame
                
                # Write the encoded H265 data to file
                h265Packet.getData().tofile(videoFile)

        except KeyboardInterrupt:
            print("Encoding stopped by user.")
    
    print("To view the encoded data, convert the stream file (.h265) into a video file (.mp4) using a command below:")
    print("ffmpeg -framerate 30 -i video.h265 -c copy video_h265.mp4")
    print("To convert .h265 type mp4 to .h264 (.mp4) using a command below")
    print("ffmpeg -i video_h265.mp4 -c:v libx264 -preset fast -crf 23 -c:a aac -b:a 128k video_h264.mp4")
