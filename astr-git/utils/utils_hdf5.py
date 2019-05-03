import h5py as h5
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

MINES_CASES = 'mines'
NOTMINES_CASES = 'not_mines'

def load_dataset(data):
    ''' Load in images from Hdf5 file as np.arrays.
    Dimensions of the images are (640, 1360, 3)
    '''


    f = h5.File(data,'r')
    mines = f[MINES_CASES]
    notmines = f[NOTMINES_CASES]

    mines_images = []
    notmines_images = []
    for i in mines.keys():
        img = np.array(mines[i])
        mines_images.append(img)
    for i in notmines.keys():
        img = np.array(notmines[i])
        notmines_images.append(img)


    return (mines_images, notmines_images)


### GET ALL images
def create_dataset(mines_dir,notmines_dir,h5_name):
    '''Create an Hdf5 file from a directory of images'''
    mines_path = Path(mines_dir)
    notmines_path = Path(notmines_dir)

    mines_images = mines_path.glob('*.jpg')
    notmines_images = notmines_path.glob('*.jpg')

    f = h5.File(h5_name,'a')
    mines_group = f.create_group(MINES_CASES)

    for image in mines_images:
        vector_img = cv2.imread(str(image))
        mine_name = str(image).split('/')[-1]
        mines_group.create_dataset(mine_name, data= vector_img, dtype='uint8')

    notmines_group = f.create_group(NOTMINES_CASES)

    for image in notmines_images:
        vector_img = cv2.imread(str(image))
        mine_name = str(image).split('/')[-1]
        notmines_group.create_dataset(mine_name, data= vector_img, dtype='uint8')

def store_h5(mines, notmines,h5_name):
        f = h5.File(h5_name,'a')
        mines_group = f.create_group(MINES_CASES)
        rand = 0
        for image in mines:
            mine_name = str(rand) + "-" +str(image)
            mines_group.create_dataset(mine_name, data= image, dtype='uint8')
            rand +=1
        notmines_group = f.create_group(NOTMINES_CASES)

        for image in notmines:
            mine_name = str(rand) + "-" +str(image)
            notmines_group.create_dataset(mine_name, data= image, dtype='uint8')
            rand+=1

# check = load_dataset()
