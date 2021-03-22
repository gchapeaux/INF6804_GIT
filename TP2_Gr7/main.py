import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image

from scripts.bg_sub import bgs_detection, bgs_bg
from scripts.yolo import yolo_detection
from scripts.gt_bbox import gt_detection
from scripts.iou import global_iou

'''
Métrique : IoU
'''

def single_picture_analysis(path_to_query, path_to_groundtruth, path_to_video, path_to_yolo='yolo-object-detection/yolo-coco', verbose=False):
    
    ipt_img = Image.open(path_to_query)
    shape = ipt_img.size
    gt_img, gt_boxes = gt_detection(path_to_groundtruth, show=False)
    bgs_img, bgs_boxes = bgs_detection(path_to_query, path_to_video, verbose=False, show=False)
    yolo_img, yolo_boxes = yolo_detection(path_to_query, path_to_yolo, verbose=False, show=False)

    bgs_iou = global_iou(shape, gt_boxes, bgs_boxes)
    yolo_iou = global_iou(shape, gt_boxes, yolo_boxes)

    if verbose:
        print("IoU metric")
        print("> Background substraction : "+str(bgs_iou)[0:4])
        print("> YOLO detection : "+str(yolo_iou)[0:4])

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

def sixchar(int):
    string = '000000'+str(int)
    return string[-6:]

def video_analysis(query_gt_paths, path_to_video, detection, path_to_yolo='yolo-object-detection/yolo-coco', verbose=False):

    if detection == 'bgs':
        if verbose: print('Processing Background substraction detection on video')
        bg = bgs_bg(path_to_video)
    elif detection == 'yolo':
        if verbose: print('Processing Yolo detection on video')

    ious = []

    for input_path, gt_path in query_gt_paths:
        
        ipt_img = Image.open(input_path)
        shape = ipt_img.size
        gt_img, gt_boxes = gt_detection(gt_path, show=False)
        if detection == 'bgs':            
            img, boxes = bgs_detection(input_path, path_to_video, bg=bg, verbose=False, show=False)
        elif detection == 'yolo':
            img, boxes = yolo_detection(input_path, path_to_yolo, verbose=False, show=False)

        iou = global_iou(shape, gt_boxes, boxes)
        ious.append(iou)

    return np.mean(ious)

def main(H0=True, H1=True, H2=True, H3=True):

    if not(os.path.exists('./data') and os.path.isdir('./data')):
        os.mkdir('./data')
    if not os.listdir('./data'):
        print("Aucune donnée trouvée. Téléchargez les dataset suivants et extrayez les dans le dossier ./data")
        print("| PETS2006 : http://jacarini.dinf.usherbrooke.ca/static/dataset/baseline/PETS2006.zip")
        print("| canoe : http://jacarini.dinf.usherbrooke.ca/static/dataset/dynamicBackground/canoe.zip")
        print("| busStation : http://jacarini.dinf.usherbrooke.ca/static/dataset/shadow/busStation.zip")
        print("| sofa : http://jacarini.dinf.usherbrooke.ca/static/dataset/intermittentObjectMotion/sofa.zip")
        return

    if H0:
        print("\n= Cas de base : PETS2006 =\n")
        path_to_video = 'data/PETS2006/input'

        path_to_query = 'data/PETS2006/input/in000987.jpg'
        path_to_groundtruth = 'data/PETS2006/groundtruth/gt000987.png'
        single_picture_analysis(path_to_query, path_to_groundtruth, path_to_video)

        query_gt_paths = list(map(lambda x, y : (os.path.join('data/PETS2006/input', x), os.path.join('data/PETS2006/groundtruth', y)), os.listdir('data/PETS2006/input'), os.listdir('data/PETS2006/groundtruth')))
        print("GIoU pour la soustraction d'arrière plan : "+str(video_analysis(query_gt_paths[800:1000], path_to_video, 'bgs'))[0:4])
        print("GIoU pour la détection Yolo : "+str(video_analysis(query_gt_paths[800:1000], path_to_video, 'yolo'))[0:4])

    if H1:
        print("\n= Arrière-plan dynamique : canoe =\n")
        path_to_video = 'data/canoe/input'

        path_to_query = 'data/canoe/input/in000987.jpg'
        path_to_groundtruth = 'data/canoe/groundtruth/gt000987.png'
        single_picture_analysis(path_to_query, path_to_groundtruth, path_to_video)

        query_gt_paths = list(map(lambda x, y : (os.path.join('data/canoe/input', x), os.path.join('data/canoe/groundtruth', y)), os.listdir('data/canoe/input'), os.listdir('data/canoe/groundtruth')))
        print("GIoU pour la soustraction d'arrière plan : "+str(video_analysis(query_gt_paths[800:1000], path_to_video, 'bgs'))[0:4])
        print("GIoU pour la détection Yolo : "+str(video_analysis(query_gt_paths[800:1000], path_to_video, 'yolo'))[0:4])

    if H2:
        print("\n= Détection des ombres : busStation =\n")
        path_to_video = 'data/busStation/input'

        path_to_query = 'data/busStation/input/in000987.jpg'
        path_to_groundtruth = 'data/busStation/groundtruth/gt000987.png'
        single_picture_analysis(path_to_query, path_to_groundtruth, path_to_video)

        
        query_gt_paths = list(map(lambda x, y : (os.path.join('data/busStation/input', x), os.path.join('data/busStation/groundtruth', y)), os.listdir('data/busStation/input'), os.listdir('data/busStation/groundtruth')))
        print("GIoU pour la soustraction d'arrière plan : "+str(video_analysis(query_gt_paths[900:1100], path_to_video, 'bgs'))[0:4])
        print("GIoU pour la détection Yolo : "+str(video_analysis(query_gt_paths[900:1100], path_to_video, 'yolo'))[0:4])

    if H3:
        print("\n= Détection avec occlusion : sofa =\n")
        path_to_video = 'data/sofa/input'

        path_to_query = 'data/sofa/input/in001995.jpg'
        path_to_groundtruth = 'data/sofa/groundtruth/gt001995.png'
        single_picture_analysis(path_to_query, path_to_groundtruth, path_to_video)

        query_gt_paths = list(map(lambda x, y : (os.path.join('data/sofa/input', x), os.path.join('data/sofa/groundtruth', y)), os.listdir('data/sofa/input'), os.listdir('data/sofa/groundtruth')))
        print("GIoU pour la soustraction d'arrière plan : "+str(video_analysis(query_gt_paths[1950:2150], path_to_video, 'bgs'))[0:4])
        print("GIoU pour la détection Yolo : "+str(video_analysis(query_gt_paths[1950:2150], path_to_video, 'yolo'))[0:4])
        
main()