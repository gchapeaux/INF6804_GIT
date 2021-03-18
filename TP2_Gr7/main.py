import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image

from scripts.bg_sub import bgs_detection
from scripts.yolo import yolo_detection
from scripts.gt_bbox import gt_detection
from scripts.iou import global_iou

'''
Métrique : IoU
'''

def main(path_to_data, query_id):
    
    # PATH TO THE INPUT DATA
    input_path = path_to_data+'input/in'+query_id+'.jpg'
    gt_path = path_to_data+'groundtruth/gt'+query_id+'.png'

    # PATH TO THE YOLO RECOGNITION SYSTEM 
    yolo_path = 'yolo-object-detection/yolo-coco'

    # PATH TO VIDEO
    video_path = path_to_data+"input/"

    ipt_img = Image.open(input_path)
    shape = ipt_img.size
    gt_img, gt_boxes = gt_detection(gt_path, show=False)
    bgs_img, bgs_boxes = bgs_detection(input_path, video_path, show=False)
    yolo_img, yolo_boxes = yolo_detection(yolo_path, input_path, verbose=False, show=False)

    bgs_iou = global_iou(shape, gt_boxes, bgs_boxes)
    yolo_iou = global_iou(shape, gt_boxes, yolo_boxes)

    print("IoU metric")
    print("> Background substraction : "+str(bgs_iou))
    print("> YOLO detection : "+str(yolo_iou))

    fig, ((ipt, gt), (bgs, yolo)) = plt.subplots(nrows=2, ncols=2)

    ipt.imshow(ipt_img)
    ipt.axis('off')
    ipt.set_title('Input picture')

    gt.imshow(gt_img)
    gt.axis('off')
    gt.set_title('Groundtruth')

    bgs.imshow(bgs_img)
    bgs.axis('off')
    bgs.set_title('Background substraction')

    yolo.imshow(yolo_img)
    yolo.axis('off')
    yolo.set_title('YOLO description')

    plt.show()


# PATH TO THE VIDEO
data_path = 'data/PETS2006/'

# ID OF THE INPUT PICTURE
query_id = '000987'

main(data_path, query_id)