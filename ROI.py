import cv2
import os
import numpy as np


def split_into_chunks(arr, chunk_size):
    return np.array_split(arr, len(arr) // chunk_size)


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Mouse clicked at: ({y}, {x})")


ArrayOfCoordinates = (np.genfromtxt('ManCoord.txt'))
ArrayOfCoordinates = ArrayOfCoordinates.astype(np.int64, copy=False)

originalImage = cv2.imread('OnClickImg.jpg')
# dim = (1366, 768)
# resizedImage = cv2.resize(originalImage, dim)

Chunks = split_into_chunks(ArrayOfCoordinates, 4)

ArrayOfChunks = np.array(Chunks)

for coord in ArrayOfChunks:
    cv2.line(originalImage, (coord[0], coord[1]), (coord[2], coord[3]), (255, 0, 0), 2)

# Attach the callback function to the window
cv2.namedWindow('Original Image')
cv2.setMouseCallback('Original Image', mouse_callback)

slicedImage = originalImage[65:626, 855:1166]  # y1:y2, x1:x2

cv2.imshow("Original Image", originalImage)
cv2.imshow("Sliced Image", slicedImage)
# cv2.imshow("Sliced Image", resizedImage)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Mouse clicked at: 95:695, 610:1061
# (434, 487) (703, 837)
# (453, 694) (779, 1048)
# (545, 708) (716, 1045)
# (443, 256) (697, 627)
# (73, 837) (600, 1126)
# (65, 855) (626, 1166)
