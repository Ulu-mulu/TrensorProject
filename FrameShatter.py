import cv2

capture = cv2.VideoCapture('C:/Users/Delain/PycharmProjects/YoloTorch/RawVideos/Alex.avi')

FrameNum = 0
FrameName = 'Alex'

while True:

    success, frame = capture.read()

    if success:

        sliceOfImage = frame[65:626, 855:1166]

        cv2.imwrite(f'C:/Users/Delain/PycharmProjects/YoloTorch/Alex/train/images/{FrameName}_{FrameNum}.jpg', sliceOfImage)

    else:
        break

    FrameNum = FrameNum + 1

capture.release()
