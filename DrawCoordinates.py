import os
import cv2
import numpy as np
from cv2 import VideoCapture


def split_into_chunks(arr, chunk_size):
    return np.array_split(arr, len(arr) // chunk_size)


originalImage = cv2.imread('ShatteredFrames/Me_0.jpg')
ArrayOfCoordinates = (np.genfromtxt('ManCoord.txt'))
ArrayOfCoordinates = ArrayOfCoordinates.astype(np.int64, copy=False)

Chunks = split_into_chunks(ArrayOfCoordinates, 4)

ArrayOfChunks = np.array(Chunks)

for coord in ArrayOfChunks:
    cv2.line(originalImage, (coord[0], coord[1]), (coord[2], coord[3]), (255, 0, 0), 2)

cv2.imwrite("LabeledImage.jpg", originalImage)
