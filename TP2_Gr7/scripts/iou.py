import numpy as np
import cv2


def rel_to_abs_box(box):
    x, y, w, h = box[0], box[1], box[2], box[3]
    return x, y, x+w, y+h

'''
SOURCE : http://ronny.rest/tutorials/module/localization_001/iou/
'''
def get_iou(boxa, boxb, epsilon=1e-5):
    """ 
    Args:
        boxa:          (list of 4 numbers) [xa, ya, wa, ha]
        boxb:          (list of 4 numbers) [xb, yb, wb, hb]
        epsilon:       (float) Small value to prevent division by zero

    Returns:
        (float) The Intersect of Union score.
    """

    xa1, ya1, xa2, ya2 = rel_to_abs_box(boxa)
    xb1, yb1, xb2, yb2 = rel_to_abs_box(boxb)

    # COORDINATES OF THE INTERSECTION BOX
    x1 = max(xa1, xb1)
    y1 = max(ya1, yb1)
    x2 = min(xa2, xb2)
    y2 = min(ya2, yb2)

    # AREA OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)
    # handle case where there is NO overlap
    if (width<0) or (height <0):
        return 0.0
    area_overlap = width * height

    # COMBINED AREA
    area_a = (xa2-xa1) * (ya2-ya1)
    area_b = (xb2-xb1) * (yb2-yb1)
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / max(area_combined, epsilon)
    return iou

def global_iou(gt_boxes, det_boxes):
    ious = []
    corr_boxes = []
    for gt_box in gt_boxes:
        corr_iou = -1.0
        corr_box = None
        for det_box in det_boxes:
            det_iou = get_iou(gt_box, det_box)
            if det_iou > corr_iou:
                corr_iou = det_iou
                corr_box = det_box
        ious.append(corr_iou)
        corr_boxes.append(corr_box)
    giou = np.mean(ious)
    return giou