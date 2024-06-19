import cv2
import os
import numpy as np
from cv2 import VideoCapture
import torch
from ultralytics import YOLO
import threading
import time
from time import sleep
import logging


class RepeatedTimer(object):
    def __init__(self, first_interval, interval, func, *args, **kwargs):
        self.timer = None
        self.first_interval = first_interval
        self.interval = interval
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.running = False
        self.is_started = False

    def first_start(self):
        try:
            # no race-condition here because only control thread will call this method
            # if already started will not start again
            if not self.is_started:
                self.is_started = True
                self.timer = threading.Timer(self.first_interval, self.run)
                self.running = True
                self.timer.start()
        except Exception as e:
            log_print(syslog.LOG_ERR, "timer first_start failed %s %s" % (e.message, traceback.format_exc()))
            raise

    def run(self):
        # if not stopped start again
        if self.running:
            self.timer = threading.Timer(self.interval, self.run)
            self.timer.start()
        self.func(*self.args, **self.kwargs)

    def stop(self):
        # cancel current timer in case failed it's still OK
        # if already stopped doesn't matter to stop again
        if self.timer:
            self.timer.cancel()
        self.running = False


def predict_image(frame, model):
    sliced_image = frame #[95:695, 610:1061]
    results = model(sliced_image)
    cv2.imshow('Region of interest', sliced_image)


rtsp_username = "admin"
rtsp_password = "Qwerty1234"
width = 1920
height = 1080
cam_no = 1
fps = 25

os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'

rtsp: str = "rtsp://" + rtsp_username + ":" + rtsp_password + "@10.200.1.45:554/Streaming/channels/" + "1" + "01"
cap: VideoCapture = cv2.VideoCapture()
cap.open(rtsp)
cap.set(3, 1920)  # ID number for width is 3
cap.set(4, 1080)  # ID number for height is 480
cap.set(10, 100)  # ID number for brightness is 10qq

# initialize model
model = YOLO('yolov8m.pt')
model.to('cuda')

while True:

    ret, frame = cap.read()

    repeat = RepeatedTimer(10.0, 300000.0, predict_image, frame, model)
    # it auto-starts, no need of rt.start().

    if ret:
        if not repeat.running:
            repeat.run()

        # Display the resulting frame
        cv2.imshow('frame', frame)

        # Press Q on keyboard to stop recording
        if cv2.waitKey(1) & 0xFF == ord('q'):
            repeat.stop()
            break

    # Break the loop
    else:
        repeat.stop()
        break

if repeat.running:
    repeat.stop()

# When everything done, release the video capture and video write objects
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
