from images_utils import fetch_accross_states, crop_images
from utils_hdf5 import create_dataset, load_dataset

from PIL import Image
import cv2



MINES_RAW = '../raw/mines'
NOTMINES_RAW = '../raw/not_mines'

MINES_CLEAN = '../clean/mines'
NOTMINES_CLEAN = '../clean/not_mines'

nev_coord = [-10104546,1607629,-10098522,1612354]
cali_coord = [-13296940,5085537,-13288245,5090262]
northdk_coord = [-11247752,6030341,-11241728,6035066]
colorado_coord = [-11769378,4792324,-11763353,4797049]

STATES_COORDINATES = [nev_coord,cali_coord,northdk_coord]

# fetch_accross_states(STATES_COORDINATES,NOTMINES_RAW)
crop_images(MINES_RAW,MINES_CLEAN)
crop_images(NOTMINES_RAW,NOTMINES_CLEAN)

h5_filename = "dataset.hdf5"
create_dataset(MINES_CLEAN,NOTMINES_CLEAN,h5_filename)
load_dataset(h5_filename)
