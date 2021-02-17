from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import json

#Region[Red] 1. Regions of interest identification

def draw_ROI(box, img, cover, width=2,color='red'):
    draw = ImageDraw.Draw(img)
    l, c, dl, dc = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    if cover:
        draw.rectangle((l, c, l+dl, c+dc), fill=color, outline=color)
    else:
        draw.line([(l,c),(l+dl,c)], fill=color, width=width)
        draw.line([(l,c),(l,c+dc)], fill=color, width=width)
        draw.line([(l+dl,c),(l+dl,c+dc)], fill=color, width=width)
        draw.line([(l,c+dc),(l+dl,c+dc)], fill=color, width=width)

def regions_of_interest(file_name, saving=False):
    with open('data/part1/gt.json', 'r') as file:
        data = json.loads(file.read())
    path = './data/part1/images/'+file_name+'.jpg'
    image = Image.open(path)
    for zone in data['annotations']:
        if zone['image'] == file_name:
            draw_ROI(zone['bbox'], image, zone['category_id'] == 8)
    image.show()
    if saving:
        image.save("output.png")

#EndRegion