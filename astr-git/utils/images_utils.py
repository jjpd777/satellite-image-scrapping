import cv2
import shutil
import requests
from pathlib import Path
import random


random.seed(111)

### Cropping images:
def crop_images(origin_dir,destination_dir):

    path = Path(origin_dir)
    mines = path.glob('*.jpg')

    for image in mines:
        print(image)
        im = cv2.imread(str(image))
        # print(im)
        img = im[0:580,200:1180].copy()

        name = str(image).split('/')[-1]
        path_to_image = destination_dir + name

        cv2.imwrite(path_to_image,img)


# crop_images(directory,destination)


def create_new_coordinates(original_box,num):
    acc = ''
    for i in range(len(original_box)):
        new_box = original_box[i] + num
        acc+=str(new_box) + ','
    acc = acc[:-1]
    return acc

def create_tiles(coordinates,number_of_samples=20):
    boxes = []
    for i in range(number_of_samples):
        shift_direction = random.choice([(50000,100000),(-100000,-50000)])
        coordinate_shift = random.randint(shift_direction[0],shift_direction[1])

        coord = create_new_coordinates(coordinates,coordinate_shift)
        boxes.append(coord)

    return boxes



BASE_URL ='''https://services.sentinel-hub.com/ogc/wms/b7b5e3ef-5a40-4e2a-9fd3-75ca2b81cb32?SERVICE=WMS&amp&REQUEST=GetMap&amp&MAXCC=20&amp&LAYERS=1-NATURAL-COLOR&amp&EVALSOURCE=S2&amp&WIDTH=1281&amp&HEIGHT=989&amp&FORMAT=image/jpeg&amp&NICENAME=Sentinel-2+image+on+2019-04-29.jpg&amp&TIME=2018-10-01/2019-04-29&amp&BBOX='''

def fetch_images(coordinates,output_dir):
    boxes = create_tiles(coordinates,100)
    for box in boxes:
        url = BASE_URL + box
        r = requests.get(url)
        count = 'image'+ str(box)
        name = output_dir +count+'random-'+box[1:5]+'.jpg'
        print(r.status_code)
        if r.status_code == 200:
            with open(name, 'wb') as f:
                print(name)
                for chunk in r.iter_content(1024):
                    f.write(chunk)
                f.close()


def fetch_accross_states(list_of_states, output_dir):
    for coordinates  in range(len(list_of_states)):
        fetch_images(list_of_states[coordinates], output_dir)
