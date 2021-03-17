import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

'''
    SOURCE : https://stackoverflow.com/questions/60646384/python-opencv-background-subtraction-and-bounding-box
'''

#Region[Red] Parameters

EROSION_SIZE = 5
DILATATION_SIZE = 20
BG_SENSITIVITY = 15

#EndRegion

#Region[Yellow] Functions 

# Generate background from the first 300 frames of the video
def bgs_bg(video_path):
    nb_img = 0
    moy = None
    var = None
    for file in os.listdir(video_path):
        if file.endswith(".jpg") or file.endswith(".png"):
            nb_img += 1  
            image = (cv2.imread(os.path.join(video_path, file),cv2.IMREAD_GRAYSCALE)).astype(float)
            if moy is None:
                moy = image
                var = np.power(image, 2)
            else:
                moy = moy + image
                var = var + np.power(image, 2)
    if moy is None:
        raise Exception('No image in the directory')
    moy = moy/nb_img
    var = var/nb_img - np.power(moy, 2)
    return moy, var

# Compute foreground
def bgs_fg(query_path, background, s=BG_SENSITIVITY):
    moy, var = background
    image_query = (cv2.imread(query_path,cv2.IMREAD_GRAYSCALE)).astype(float)
    #fg = np.abs(image_query-moy) > s*np.sqrt(var) 
    fg = np.abs(image_query-moy) > s 
    return fg

# Process bounding boxes
def bgs_detection(query_path, video_path, bg=None, erosion_size = EROSION_SIZE, dilatation_size = DILATATION_SIZE, show=True):
    
    if bg is None:
        bg = bgs_bg(video_path)
    fg = bgs_fg(query_path, bg)

    bnw_img = fg.astype(np.float32)
    img = cv2.cvtColor(bnw_img, cv2.COLOR_GRAY2BGR)
    
    # apply erode
    erosion_type = cv2.MORPH_ELLIPSE
    element = cv2.getStructuringElement(erosion_type, (2*erosion_size + 1, 2*erosion_size+1), (erosion_size, erosion_size))
    img_erosion = cv2.erode(bnw_img, element)
    # apply dilate
    dilatation_type = cv2.MORPH_ELLIPSE
    element = cv2.getStructuringElement(dilatation_type, (2*dilatation_size + 1, 2*dilatation_size+1), (dilatation_size, dilatation_size))
    img_dilate = cv2.dilate(img_erosion, element).astype(np.uint8)

    contours, hierarchy = cv2.findContours(img_dilate, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        boxes.append( [x, y, int(w), int(h)] )

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if show:
        plt.imshow(img)
        plt.show()

    return img, boxes

#EndRegion