import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

videopath = 'data/PETS2006/input/'

def generate_bg(videopath):
    nb_img = 0
    background = None
    for file in os.listdir(videopath):
        if file.endswith(".jpg") or file.endswith(".png"):
            nb_img += 1  
            image = (cv2.imread(os.path.join(videopath, file),cv2.IMREAD_COLOR)).astype(float)
            if background is None:
                background = image
            else:
                background = background + image
    if background is None:
        raise Exception('No image in the directory')
    return background/nb_img

bg = generate_bg(videopath)

plt.imshow(bg, cmap = plt.get_cmap('gray'))
plt.show()

