import matplotlib.pyplot as plt
import cv2
from utils.utils_hdf5 import load_dataset, store_h5
import imgaug.augmenters as iaa
import numpy as np
import random
random.seed(777)



def reshape(img):
   im = cv2.resize(img,dsize=(int(img.shape[1]/5),int(img.shape[0]/5)), interpolation = cv2.INTER_NEAREST)
   color_format =cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
   return color_format

def peak_images_plot(dataset_name,output_png):
    data = load_dataset(dataset_name)
    mines_data = data[0]
    notmines_data = data[1]

    print("Both set of images are of the same shape:")
    print(mines_data[0].shape, notmines_data[0].shape)

    f, ax = plt.subplots(2,5, figsize=(30,10))
    for i in range(10):
        if i<5:
            img = reshape(mines_data[i])
            name = "mine-number-" +str(i)
        else:
            img = reshape(notmines_data[i])
            name = "notmine-number-" +str(i)

        ax[i//5, i%5].imshow(img)
        ax[i//5, i%5].set_title(name)
        ax[i//5, i%5].axis('off')
        ax[i//5, i%5].set_aspect('auto')

    plt.savefig(output_png)


#### Create more samples of underepresented class
def increase_underrepresented(underepresented):
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


def add_labels(positive, negative):
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
    FEATURE = np.array(FEATURE)/255
    LABEL = np.array(LABEL)

    return [FEATURE,LABEL]
