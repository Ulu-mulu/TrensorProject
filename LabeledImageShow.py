import os
import cv2
import numpy as np
from cv2 import VideoCapture
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from PIL import Image, ImageDraw


def split_into_chunks(arr, chunk_size):
    return np.array_split(arr, len(arr) // chunk_size)


image = Image.open('icons/TB/philips-243v5lhsb00_220389__1_normal_extra.jpg')
draw = ImageDraw.Draw(image)

# originalImage = cv2.imread('ShatteredFrames/Me_0.jpg')
ArrayOfCoordinates = (np.genfromtxt('ManCoord.txt'))
ArrayOfCoordinates = ArrayOfCoordinates.astype(np.int64, copy=False)

Chunks = split_into_chunks(ArrayOfCoordinates, 4)

ArrayOfChunks = np.array(Chunks)

for coord in ArrayOfChunks:
    # cv2.line(originalImage, (coord[0], coord[1]), (coord[2], coord[3]), (255, 0, 0), 2)
    draw.line((coord[0], coord[1], coord[2], coord[3]), fill='yellow', width=4)

image.save('labeledTV.jpg')
# cv2.imshow("Labeled Image", originalImage)

# cv2.waitKey(0)
# cv2.destroyAllWindows()
