import cv2

import numpy as np
from matplotlib import pyplot as plt
import itertools
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import preprocess
import pickle 
import keras

# creating dictionary for labels
class_labels = {str(x):x for x in range(10)}
class_labels.update({'+':10, 'times':11, '-':12 })
label_class = dict( zip(class_labels.values(), class_labels.keys() ))

path1 = './Equation data/'
temp1 = np.load(path1+'test9.npy')
temp2 = np.load(path1+'Equations_images_2.npy')
temp3 = np.load(path1+'Equations_images_3.npy')
temp4 = np.load(path1+'Equations_images_4.npy')
#temp5 = np.load(path1+'/test3.npy')
eqn_full = [temp1]

# Loading saved models for prediction
#path = './trained models/'
path = './trained models/'

adastage1_ver1,adadigits_ver1, adachars_ver1, rfmodel_ver1, MLP_single_ver1, cnn_ver1 = preprocess.load_models(path,1)
adastage1_ver2,adadigits_ver2, adachars_ver2, rfmodel_ver2, MLP_single_ver2, cnn_ver2 = preprocess.load_models(path,2)


#Evaluating each equation image
for eqns in eqn_full:
    for c in range(len(eqns)):
        # initialising strings to store output of model predictions
        rf_pred_ver1 = ''
        ada_pred_ver1 = ''
        mlp_pred_ver1 = ''
        rf_pred_ver2 = ''
        ada_pred_ver2 = ''
        mlp_pred_ver2 = ''
        cnn_pred_ver1 = ''
        cnn_pred_ver2 = ''

        #         print('\nEquation = ',c)
        eqn1 = eqns[c]
        # extract segments (digits/symbols) from each equation image
        segments = preprocess.extract_segments(eqn1, 30, reshape=1, size=[28, 28],
                                               area=100, gray=True, dil=True, ker=1)

        # run prediction on each segment
        plt.figure(figsize=(10, 5))
        for i in range(len(segments) + 1):
            if i == 0:
                plt.subplot(191)
                plt.imshow(eqn1, cmap='gray')

            else:
                # plot each segment
                #plt.subplot(191 + i)
                temp = segments[i - 1]
                plt.imshow(temp, cmap='gray')
                temp = temp.reshape(1, -1)

                pred = preprocess.predict(temp, label_class, 0, adastage1_ver1, adadigits_ver1, adachars_ver1,
                                          rfmodel_ver1, MLP_single_ver1, cnn_ver1)

                ada_pred_ver1 += pred[0]
                rf_pred_ver1 += pred[1]
                mlp_pred_ver1 += pred[2]
                cnn_pred_ver1 += pred[3]

                pred = preprocess.predict(temp, label_class, 0, adastage1_ver2, adadigits_ver2, adachars_ver2,
                                          rfmodel_ver2, MLP_single_ver2, cnn_ver2)

                ada_pred_ver2 += pred[0]
                rf_pred_ver2 += pred[1]
                mlp_pred_ver2 += pred[2]
                cnn_pred_ver2 += pred[3]

        plt.show()

        #print('Recognized Expression = ', cnn_pred_ver2)
b=''
for i in range(len(cnn_pred_ver2)):
    if (cnn_pred_ver2[i] == 't'):
        b = b + '*'
    elif(cnn_pred_ver2[i]==' '):
        pass
    elif(cnn_pred_ver2[i]=='i'):
       pass
    elif (cnn_pred_ver2[i] == 'm'):
        pass
    elif (cnn_pred_ver2[i] == 'e'):
        pass
    elif (cnn_pred_ver2[i] == 's'):
        pass
    elif(cnn_pred_ver2[i]!=' '):
        b = b + cnn_pred_ver2[i]
print('Recognized Expression = ', b)

print("\nResult =",eval(b))