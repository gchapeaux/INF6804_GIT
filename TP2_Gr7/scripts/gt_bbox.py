import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def gt_detection(path_to_query, show=True):

    image_query = (cv2.imread(path_to_query, cv2.IMREAD_GRAYSCALE))

    contours, hierarchy = cv2.findContours(image_query, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    img = cv2.cvtColor(image_query, cv2.COLOR_GRAY2RGB)
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        boxes.append( [x, y, int(w), int(h)] )

    if show:
        plt.imshow(img)
        plt.show()

    return img, boxes

