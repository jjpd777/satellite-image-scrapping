from utils.sentinel_scraper import fetch_accross_states
from utils.utils_hdf5 import create_dataset, load_dataset
from utils.image_utils import crop_images

from PIL import Image
import cv2



MINES_RAW = 'dataset/raw/mines/'
NOTMINES_RAW = 'dataset/raw/trial/'

MINES_CLEAN = 'dataset/clean/mines'
NOTMINES_CLEAN = 'dataset/clean/not_mines/'
H5_FILENAME = "dataset.hdf5"

TEST_DATA = "dataset/raw/test/"
TEST_DATA_CLEAN = "dataset/clean/test/"


nev_coord = [-10104546,1607629,-10098522,1612354]
cali_coord = [-13296940,5085537,-13288245,5090262]
northdk_coord = [-11247752,6030341,-11241728,6035066]

colorado_coord = [-11769378,4792324,-11763353,4797049]
texas_coord = [-11125195,3757286,-11119405,3761776]
south_dk_coord = [-8575313,4710855,-8572194,4715346]

STATES_COORDINATES = [nev_coord,cali_coord,northdk_coord]
TEST_STATES = [colorado_coord,cali_coord,south_dk_coord]
'''Only fetching for tiles with no mines. Mines images were scraped manually
   using the Sentinel-hub playground browser.'''
# fetch_accross_states(STATES_COORDINATES,NOTMINES_RAW)
# fetch_accross_states(TEST_STATES,TEST_DATA)
'''Once images are scraped, crop them and save them to new directory'''
# crop_images(MINES_RAW,MINES_CLEAN)
# crop_images(NOTMINES_RAW,NOTMINES_CLEAN)
# crop_images(TEST_DATA, TEST_DATA_CLEAN)
KEYS = ['mines','notmines','test_cases']
PATHS = [MINES_CLEAN, NOTMINES_CLEAN, TEST_DATA_CLEAN]


create_dataset(KEYS,PATHS,H5_FILENAME)
# X,Y,T,_ = load_dataset(H5_FILENAME)
# print(len(T))
# print(len(X))
# print(len(Y))
