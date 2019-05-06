import cv2
import shutil
import requests
from pathlib import Path
import random


random.seed(111)

def shift_box(original_box,random_num):
    '''From a sample BBox, create new coordinates by adding or substracting
     a random number in either direction of the XY coordinate plane'''
    acc = ''
    for i in range(len(original_box)):
        new_box = original_box[i] + random_num
        acc+=str(new_box) + ','
    acc = acc[:-1]
    return acc

def create_multiple_boxes(original_box,number_of_samples):
    '''Given a BBox set of coordinates, generate
        more boxes of nearby locations'''
    boxes = []
    for i in range(number_of_samples):
        shift_direction = random.choice([(50000,100000),(-100000,-50000)])
        coordinate_shift = random.randint(shift_direction[0],shift_direction[1])

        box = shift_box(original_box,coordinate_shift)
        boxes.append(box)

    return boxes

BASE_URL ='''https://services.sentinel-hub.com/ogc/wms/b7b5e3ef-5a40-4e2a-9fd3-75ca2b81cb32?SERVICE=WMS&amp&REQUEST=GetMap&amp&MAXCC=20&amp&LAYERS=1-NATURAL-COLOR&amp&EVALSOURCE=S2&amp&WIDTH=1281&amp&HEIGHT=989&amp&FORMAT=image/jpeg&amp&NICENAME=Sentinel-2+image+on+2019-04-29.jpg&amp&TIME=2018-10-01/2019-04-29&amp&BBOX='''

def fetch_images(box,output_dir,sample_size=20):
    '''Fetch images corresponding to an original BBox set of coordinates
    and a list of generated nearby boxes. If image fetched correctly,
    write image to output_dir.'''
    boxes = create_multiple_boxes(box,sample_size)
    for box in boxes:
        url = BASE_URL + box
        r = requests.get(url)
        count = 'image'+ str(box)
        name = output_dir+str(box[1:5])+'image-scrapped.jpg'
        if r.status_code == 200:
            with open(name, 'wb') as f:
                print(name)
                for chunk in r.iter_content(1024):
                    f.write(chunk)
                f.close()


def fetch_accross_states(list_of_states, output_dir):
    '''Fetch tiles from a list of BBox coordinates, and
    write contents to output dictionary accordingly.'''
    print("Scraping images and writing them to: " + output_dir)

    for box in list_of_states:
        fetch_images(box, output_dir)
