import cv2
from cv2 import VideoCapture
import time

start_time = time.time()

rtsp_username = "admin"
rtsp_password = "Qwerty1234"
width = 1920
height = 1080
cam_no = 1
fps = 25

rtsp: str = "rtsp://" + rtsp_username + ":" + rtsp_password + "@10.200.1.51:554/Streaming/channels/" + "1" + "01"

cap: VideoCapture = cv2.VideoCapture()
cap.open(rtsp)

cap.set(3, width)  # ID number for width is 3
cap.set(4, height)  # ID number for height is 480
cap.set(10, 100)  # ID number for brightness is 10qq

output = cv2.VideoWriter("RawVideos/Test.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (width, height))
WrittenFrame = 0

while True:

    ret, frame = cap.read()

    if ret:

        if WrittenFrame != 1000:

            # Write the frame into the file 'output.avi'
            output.write(frame)
            WrittenFrame = WrittenFrame + 1

        else:
            break

# Break the loop
    else:
        break


# When everything done, release the video capture and video write objects
cap.release()
output.release()

print("--- %s seconds ---" % (time.time() - start_time))

# Closes all the frames
cv2.destroyAllWindows()
