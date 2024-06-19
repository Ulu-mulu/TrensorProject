import cv2
import glob

files = glob.glob('tvs/images/train/*.jpg')

FrameNum = 0

dim = (640, 640)

for frame in files:

    FrameName = frame
    originalImage = cv2.imread(frame)
    sliceOfImage = cv2.resize(originalImage, dim) # originalImage[65:626, 855:1166]
    # grayImage = cv2.cvtColor(sliceOfImage, cv2.COLOR_BGR2GRAY)

    isWritten = cv2.imwrite(f'tvs/images/test/{FrameName}.jpg', sliceOfImage)

    if isWritten:
        print('Image is successfully saved as file.')
        FrameNum = FrameNum + 1

