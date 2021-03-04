import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def generate_bg(video_path):
    nb_img = 0
    background = None
    for file in os.listdir(video_path):
        if file.endswith(".jpg") or file.endswith(".png"):
            nb_img += 1  
            image = (cv2.imread(os.path.join(video_path, file),cv2.IMREAD_GRAYSCALE)).astype(float)
            if background is None:
                background = image
            else:
                background = background + image
    if background is None:
        raise Exception('No image in the directory')
    return background/nb_img

def bg_sub(query_path, n=15, background=None, videopath=None):
    if background is None:
        background = generate_bg(videopath)
    image_query = (cv2.imread(query_path,cv2.IMREAD_GRAYSCALE)).astype(float)
    return np.abs(image_query-background) > n 


# PATH TO THE THE VIDEO FRAMES
video_path = 'data/PETS2006/input/'
# PATH TO THE QUERY PICTURE
query_path = 'data/PETS2006/input/in000081.jpg'

bg = generate_bg(video_path)
fg = bg_sub(query_path, background= bg)
plt.imshow(~fg, cmap = plt.get_cmap('binary'))
plt.show()

