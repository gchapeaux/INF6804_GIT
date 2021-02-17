from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

#Region[Red] RGB description of a picture

def rgb_data(image):
    b, g, r = cv2.split(image)
    rgb_data = np.array([r, g, b])
    return rgb_data

#EndRegion

#Region[Blue] Similarities between two pictures

def rgb_similarities(file1, file2, show_hist=False):
    
    img1 = cv2.imread(os.path.join(file1))
    img2 = cv2.imread(os.path.join(file2))
    size = (img1.shape[1], img1.shape[0])
    img2 = cv2.resize(img2, size, interpolation=cv2.INTER_AREA)

    rgb1 = rgb_data(img1)
    rgb2 = rgb_data(img2)
    
    dist = 1/3*(np.linalg.norm(rgb2[0]-rgb1[0])+np.linalg.norm(rgb2[1]-rgb1[1])+np.linalg.norm(rgb2[2]-rgb1[2]))
    sim = 1/(1+dist)

    #sim = np.dot(rgb1, rgb2) / np.linalg.norm(rgb1)*np.linalg.norm(rgb2)

    if show_hist:
        fig, ((hist1, pic1), (hist2, pic2)) = plt.subplots(2,2)
        fig.canvas.set_window_title('RGB histograms for '+str(file1)+' and '+str(file2)+' (normalized)')
        fig.canvas.set_window_title('RGB histograms of the picture')
        hr = cv2.calcHist([rgb1],[2],None,[256],[0,256])
        hist1.plot(hr,'r')
        hg = cv2.calcHist([rgb1],[1],None,[256],[0,256])
        hist1.plot(hg,'g')
        hb = cv2.calcHist([rgb1],[0],None,[256],[0,256])
        hist1.plot(hb,'b')
        pic1.imshow(Image.open(file1))
        pic1.axis('off')
        hr = cv2.calcHist([rgb2],[2],None,[256],[0,256])
        hist2.plot(hr,'r')
        hg = cv2.calcHist([rgb2],[1],None,[256],[0,256])
        hist2.plot(hg,'g')
        hb = cv2.calcHist([rgb2],[0],None,[256],[0,256])
        hist2.plot(hb,'b')
        pic2.imshow(Image.open(file2))
        pic2.axis('off')
        plt.show(block=True)

    return sim

#EndRegion
