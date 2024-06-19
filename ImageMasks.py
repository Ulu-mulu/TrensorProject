import cv2
from ultralytics import YOLO
import numpy as np
import torch

img = cv2.imread('ShatteredFrames/Data/Me_16.jpg')
model = YOLO('yolov8m-seg.pt')
results = model.predict(source=img.copy(), save=True, save_txt=False, stream=True)
for result in results:
    # get array results
    masks = result.masks.data
    boxes = result.boxes.data
    # extract classes
    clss = boxes[:, 5]
    # get indices of results where class is 0 (people in COCO)
    people_indices = torch.where(clss == 0)
    # use these indices to extract the relevant masks
    people_masks = masks[people_indices]
    # scale for visualizing results
    people_mask = torch.any(people_masks, dim=0).int() * 255
    # save to file
    cv2.imwrite(str(model.predictor.save_dir / 'merged_seg.txt'), people_mask.cpu().numpy())
