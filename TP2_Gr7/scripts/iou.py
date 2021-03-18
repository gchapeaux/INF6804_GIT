import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def bbox_mask(shape, boxes):
    mask = np.zeros(shape, dtype=bool)
    for box in boxes:
        x, y, w, h = box[0], box[1], box[2], box[3]
        mask[y:y+h, x:x+w]=True
    return mask

def iou(mask1, mask2):
    intersect = mask1 & mask2
    union = mask1 | mask2

    return np.sum(intersect)/np.sum(union)

def global_iou(shape, boxes1, boxes2):
    mask1 = bbox_mask(shape, boxes1)
    mask2 = bbox_mask(shape, boxes2)
    return iou(mask1, mask2)