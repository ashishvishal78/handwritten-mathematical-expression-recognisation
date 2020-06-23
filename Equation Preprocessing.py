import cv2
import numpy as np
from matplotlib import pyplot as plt
import itertools
import pandas as pd
import PIL
import tensorflow as tf
from preprocess import rescale_segment as rescale_segment
from preprocess import extract_segments as extract_segments

img = cv2.imread('C:/Users/DELL/Desktop/ee/Equation data/uu.jpeg', 0)
#plt.imshow(img, cmap='gray')
img= cv2.resize(img,(640,480))
plt.imshow(img.reshape(img.shape[0], img.shape[1]), cmap=plt.cm.Greys)
plt.show()
image = []  ## for eqn_im_1
for i in range(1):
    for j in range(1):
        '''x1 = 370*i ; x2 = x1+350; y1 = 500*j ; y2 = y1 + 500
        temp = img[x1:x2,y1:y2];'''
        x1 = 507000 * i;
        x2 = x1 + 505000;
        y1 = 70000 * j;
        y2 = y1 + 70000
        temp = img[x1:x2, y1:y2];

        kernel = np.ones([3, 3])
        temp = cv2.erode(temp, kernel, iterations=0)
        image.append(temp)
        plt.imshow(temp.reshape(temp.shape[0], temp.shape[1]), cmap=plt.cm.Greys)
        plt.show()

for i in range(len(image)):
    im1 = image[i]
    segments = extract_segments(im1, 30, reshape=1, size=[28, 28],
                                threshold=50, area=100, ker=1, gray=False)
    plt.figure(figsize=[15, 15])
    plt.subplot(999)
    #plt.imshow(im1, cmap='gray')
    plt.imshow(im1.reshape(temp.shape[0], im1.shape[1]), cmap=plt.cm.Greys)

    for j in range(len(segments)):

        #plt.subplot(982 + j)
        plt.imshow(segments[j], cmap='gray')
        #plt.imshow(segments[j].reshape(segments[j].shape[0], segments[j].shape[1]), cmap=plt.cm.Greys)
        plt.show()
# Saving each equation as numpy file
np.save('C:/Users/DELL/Desktop/ee/Equation data/uu.npy', np.array(image))
