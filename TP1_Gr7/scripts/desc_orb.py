import cv2
import os
import matplotlib.pyplot as plt
import imutils as imu
import numpy as np

#Region[Blue] Similarities between two pictures

def orb_similarities(file1, file2, show_matches = False): # return similarity (matching rate) between iPenvmg1 and img2 
    method = 'ORB'  # 'SIFT'
    lowe_ratio = 0.89

    img1 = cv2.imread(os.path.join(file1),cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(os.path.join(file2),cv2.IMREAD_GRAYSCALE)
    size = (img1.shape[1], img1.shape[0])
    img2 = cv2.resize(img2, size, interpolation=cv2.INTER_AREA)

    # -- Traitement ---
    # Find keypoints with ORB
    finder = cv2.ORB_create()
    kp1, des1 = finder.detectAndCompute(img1,None)
    kp2, des2 = finder.detectAndCompute(img2,None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1,des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    result_img = imu.opencv2matplotlib(cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2))

    # Compute matching rate
    mr = 1/(1+np.mean(list(m.distance for m in matches)))

    if (show_matches):
        
        fig, ax = plt.subplots(ncols=1)
        ax.imshow(result_img)
        ax.set_title('Query_img vs. database_img')
        plt.show()
        '''
        fig, ax = plt.subplots()
        im=ax.imshow(result_img)
        tit=ax.set_title ('Query_img vs. database_img')
        '''

    return mr

#EndRegion