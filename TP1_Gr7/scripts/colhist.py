from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import json
import cv2

#Region[Yellow] HSV/RGB Histogram generation

def hist_data(fpath, box, conv):
    l, c, dl, dc = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    data = cv2.imread(fpath)[l:l+dl,c:c+dc]
    if conv == 'RGB':
        return data 
    elif conv=='HSV':
        return cv2.cvtColor(data, cv2.COLOR_BGR2HSV)

def rgb_hist(fpath, box): #Pass this function as draw_hist parameter in analysis_ROI for RGB analysis
    l, c, dl, dc = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    rgb_data = hist_data(fpath, box, 'RGB')
    fig, (pic, hist) = plt.subplots(1,2)
    image = Image.open(fpath)
    pic.imshow(image.crop((l,c,l+dl,c+dc)))
    pic.axis('off')
    fig.canvas.set_window_title('RGB histograms of the picture')
    hr = cv2.calcHist([rgb_data],[2],None,[256],[0,256])
    hist.plot(hr,'r')
    hg = cv2.calcHist([rgb_data],[1],None,[256],[0,256])
    hist.plot(hg,'g')
    hb = cv2.calcHist([rgb_data],[0],None,[256],[0,256])
    hist.plot(hb,'b')
    plt.show()

def hsv_hist(fpath, box): #Pass this function as draw_hist parameter in analysis_ROI for HSV analysis
    l, c, dl, dc = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    hsv_data = hist_data(fpath, box, 'HSV')
    fig, (hist, pic) = plt.subplots(1,2)
    image = Image.open(fpath)
    pic.imshow(image.crop((l,c,l+dl,c+dc)))
    pic.axis('off')
    fig.canvas.set_window_title('HSV histograms of the picture')
    hh = cv2.calcHist([hsv_data],[0],None,[256],[0,256])
    hist.plot(hh,'salmon')
    hs = cv2.calcHist([hsv_data],[1],None,[256],[0,256])
    hist.plot(hs,'teal')
    hv = cv2.calcHist([hsv_data],[2],None,[256],[0,256])
    hist.plot(hv,'goldenrod')
    plt.show()

#EndRegion

#Region[DGray] 

def analysis_ROI(file_name, json_file, ROI_category, analysis_type, block=True):
    with open(json_file, 'r') as file:
        data = json.loads(file.read())

    id_cat = -1
    for category in data['categories']:
        if category['name'] == ROI_category:
            id_cat = category['id']
            break
    if analysis_type == 'rgb':
        draw_hist = rgb_hist
    elif analysis_type == 'hsv':
        draw_hist = hsv_hist
    else:
        raise Exception("Incorrect analysis type : "+str(analysis_type)+" - must be 'rgb' or 'hsv'")

    path = './data/part1/images/'+file_name+'.jpg'
    for zone in data['annotations']:
        if zone['image'] == file_name and zone['category_id'] == id_cat:
            print("J'ai une zone")
            draw_hist(path,zone['bbox'])
    

#EndRegion