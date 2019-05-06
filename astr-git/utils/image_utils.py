import matplotlib.pyplot as plt
import cv2
from .utils_hdf5 import load_dataset
import imgaug.augmenters as iaa
import numpy as np
import random
from pathlib import Path
random.seed(777)


def crop_images(origin_dir,destination_dir):
    ''' Get images from raw directory and write cropped images in clean directory'''
    print("Getting images from: " + origin_dir )
    print("Writing images to: " + destination_dir)

    path = Path(origin_dir)
    mines = path.glob('*.jpg')

    for image in mines:
        im = cv2.imread(str(image))
        img = im[0:580,200:1180].copy()

        name = str(image).split('/')[-1]
        path_to_image = destination_dir + name

        cv2.imwrite(path_to_image,img)

def reshape_plot(img):
    '''Reshaping images for better visualization in the matplotlib:'''
    im = cv2.resize(img,dsize=(int(img.shape[1]/5),int(img.shape[0]/5)), interpolation = cv2.INTER_NEAREST)
    color_format =cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return color_format

def peak_images_plot(dataset_name):
    '''Plotting a peak of the images in the dataset'''
    data = load_dataset(dataset_name)
    mines_data = data[0]
    notmines_data = data[1]

    print("Both set of images are of the same shape:")
    print(mines_data[0].shape, notmines_data[0].shape)

    f, ax = plt.subplots(2,5, figsize=(30,10))
    for i in range(10):
        if i<5:
            img = reshape_plot(mines_data[i])
            name = "mine-number-" +str(i)
        else:
            img = reshape_plot(notmines_data[i])
            name = "notmine-number-" +str(i)

        ax[i//5, i%5].imshow(img)
        ax[i//5, i%5].set_title(name)
        ax[i//5, i%5].axis('off')
        ax[i//5, i%5].set_aspect('auto')

def increase_underrepresented(underepresented):
    ''' Using imgaug to create additional samples of underepresented data'''
    seq = iaa.OneOf([
        iaa.Fliplr(), # horizontal flips
        iaa.Affine(rotate=20), # roatation
        iaa.Multiply((1.2, 1.5))])
    mines_augmented = []
    for image in underepresented:
        augmented1= seq.augment_image(image)
        augmented2= seq.augment_image(image)
        augmented3= seq.augment_image(image)
        mines_augmented.append(image)
        mines_augmented.append(augmented1)
        mines_augmented.append(augmented2)
        mines_augmented.append(augmented3)
    return mines_augmented
a = ['a','b']
c = ['c','d']
def create_data_feed(positive, negative):
    '''Adding labels to features, returns a list of each'''
    buffer = []
    FEATURE = []
    LABEL = []
    for i in negative:
        buffer.append([i,0])
    for i in positive:
        buffer.append([i,1])

    random.shuffle(buffer)


    for feature_label in buffer:
        FEATURE.append(feature_label[0])
        LABEL.append(feature_label[1])


    FEATURE = np.array(FEATURE)
    LABEL = np.array(LABEL)

    return [FEATURE,LABEL]
create_data_feed(a,c)

def feature_format(list_images):
    new_l = []

    for image in list_images:
        # im = cv2.resize(image,dsize=(size1,size2), interpolation = cv2.INTER_NEAREST)
        # im =cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)/255
        im = image/255
        new_l.append(im)
    new_l = np.expand_dims(new_l,axis=-1)

    return new_l
