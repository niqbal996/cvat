#!/usr/bin/env python3

import matplotlib
import matplotlib.pyplot as plt
import cv2
import numpy as np

from scipy.ndimage.morphology import binary_fill_holes

from skimage import data
from skimage.filters import try_all_threshold

from skimage.filters import threshold_otsu

from masktopolygon import convert_mask_to_polygon

def thresholdimage (img_rgb, plot=False):
    img = cv2.imread("detector/nir2_frame.png", 0)

    thresh, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

    #find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    #connectedComponentswithStats yields every seperated component with information on each of them, such as size
    #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)
    #here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    min_size = 1100  

    #your answer image
    img2 = np.zeros((output.shape))
    #for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255

    #fill holes
    img2 = binary_fill_holes(img2[:,:]).astype(int)

    img2 = np.uint8(img2)

    #apply binary mask on rgb image
    #mask3 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    #mask3 = np.broadcast_to(img2, img_rgb.shape)
    #img3 = cv2.bitwise_and(img_rgb, img_rgb, mask = img2)

    if plot:
        fig, axes = plt.subplots(ncols=5, figsize=(8, 2.5))
        ax = axes.ravel()
        ax[0] = plt.subplot(1, 5, 1, adjustable='box')
        ax[1] = plt.subplot(1, 5, 2, sharex=ax[0], sharey=ax[0], adjustable='box')
        ax[2] = plt.subplot(1, 5, 3, sharex=ax[0], sharey=ax[0], adjustable='box')
        ax[3] = plt.subplot(1, 5, 4, sharex=ax[0], sharey=ax[0], adjustable='box')
        ax[4] = plt.subplot(1, 5, 5, sharex=ax[0], sharey=ax[0], adjustable='box')

        ax[0].imshow(img_rgb, cmap=plt.cm.gray)
        ax[0].set_title('Original RGB')
        ax[0].axis('off')

        ax[1].imshow(img, cmap=plt.cm.gray)
        ax[1].set_title('Original NIR')
        ax[1].axis('off')

        #ax[1].hist(img.ravel(), bins=256)
        #ax[1].set_title('Histogram')
        #ax[1].axvline(thresh, color='r')

        ax[2].imshow(binary, cmap=plt.cm.gray)
        ax[2].set_title('Thresholded NIR image')
        ax[2].axis('off')

        ax[3].imshow(img2, cmap=plt.cm.gray)
        ax[3].set_title('region filter NIR image')
        ax[3].axis('off')

        ax[4].imshow(img3, cmap=plt.cm.gray)
        ax[4].set_title('NIR filter mask applied to rgb image')
        ax[4].axis('off')

        plt.show()
    
    return img2

if __name__ == "__main__":
    #img = cv2.imread("detector/nir2_frame.png", 0)
    img_rgb = cv2.imread("detector/rgb_frame.png")

    thresholdimage(img_rgb, False)

