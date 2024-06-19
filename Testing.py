import torch
from ultralytics import YOLO
import cv2
import numpy as np
from ultralytics.utils.plotting import Annotator

print('Pytorch CUDA Version is ', torch.version.cuda)

print('Whether CUDA is supported by our system:', torch.cuda.is_available())

cuda_id = torch.cuda.current_device()

print('CUDA Device ID: ', torch.cuda.current_device())

print('Name of the current CUDA Device: ', torch.cuda.get_device_name(cuda_id))
# initialize model
model = YOLO('TVSWeights/best.pt')
# perform inference
originalImage = cv2.imread('icons/TB/philips-288e2uae-1.jpg')
results = model(originalImage)
for r in results:
    # print(r.masks)
    originalImage = np.ascontiguousarray(originalImage)
    annotator = Annotator(originalImage)
    boxes = r.boxes
    for box in boxes:
        b = box.xyxy[0]
        c = box.cls
        annotator.box_label(b, model.names[int(c)])
        print('Object class: ', box.cls)
        print('Probability: ', box.conf)
originalImage = annotator.result()

