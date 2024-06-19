import os
import cv2
import numpy as np
from cv2 import VideoCapture
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

rtsp_username = "admin"
rtsp_password = "Qwerty1234"

rtsp: str = "rtsp://" + rtsp_username + ":" + rtsp_password + "@10.200.1.51:554/Streaming/channels/" + "1" + "01"

cap: VideoCapture = cv2.VideoCapture()
cap.open(rtsp)

cap.set(3, 1920)  # ID number for width is 3
cap.set(4, 1080)  # ID number for height is 480
cap.set(10, 100)  # ID number for brightness is 10qq

model = YOLO('MedAlexWeights/best.pt')  # initialize model
model.to('cuda')

iterate = 250

while True:
    ret, frame = cap.read()

    if ret:

        if iterate == 300:
            sliced_image = frame[65:626, 855:1166]

            # Write the frame into the model
            results = model.predict(sliced_image)
            for r in results:
                #print(r.masks)
                sliced_image = np.ascontiguousarray(sliced_image)
                annotator = Annotator(sliced_image)
                boxes = r.boxes
                for box in boxes:
                    b = box.xyxy[0]
                    c = box.cls
                    annotator.box_label(b, model.names[int(c)])
                    print('Object class: ', model.names[int(c)])
                    print('Probability: ', box.conf)

            sliced_image = annotator.result()
            cv2.imshow('Region of interest', sliced_image)

            iterate = 0

        cv2.imshow('frame', frame)

        iterate = iterate + 1

        # Press Q on keyboard to stop recording
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# When everything done, release the video capture and video write objects
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
