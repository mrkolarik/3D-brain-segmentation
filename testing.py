from __future__ import print_function

import os
import glob
import numpy as np
np.set_printoptions(threshold=np.nan)
from skimage.transform import resize
from skimage.io import imsave

from skimage.io import imread
from data import load_train_data, load_test_data


imgs_train, imgs_mask_train = load_train_data()
imgs_test = load_test_data()
imgs_mask_test = np.load('imgs_mask_test.npy')

# print(np.shape(imgs_test))
# print(np.shape(imgs_train))
# print(np.shape(imgs_mask_train))
# print(imgs_mask_train[10,11,100,:,0])

count_visualize = 1
count_processed = 0
pred_dir = 'test_final'

imgs_test = np.squeeze(imgs_test, axis=4)
if not os.path.exists(pred_dir):
    os.mkdir(pred_dir)
for x in range(0, imgs_test.shape[0]):
    for y in range(0, imgs_test.shape[1]):
        if (count_visualize > 1) and (count_visualize < 16):
            imsave(os.path.join(pred_dir, 'pred_' + str(count_processed) + '.png'), imgs_test[x][y])
            count_processed += 1
        count_visualize += 1
        if count_visualize == 17:
            count_visualize = 1
        if (count_processed % 100) == 0:
            print('Done: {0}/{1} test images'.format(count_processed, imgs_test.shape[0] * 14))

imgs_mask_test = np.squeeze(imgs_mask_test, axis=4)

imgs_test = imgs_test.astype('float32')
imgs_mask_test = imgs_mask_test.astype('float32')
# imgs_mask_test /= 1.8
# imgs_mask_test = np.around(imgs_mask_test, decimals=0)
# imgs_mask_test /= 1.75
# imgs_test /= 700.  # scale masks to [0, 1]

print(imgs_test[10,11,100,:])
print(imgs_mask_test[10,11,100,:])

# imgs_mask_train = np.around(imgs_mask_train, decimals=0)
# imgs_mask_train = np.around(imgs_mask_train, decimals=0)
# imgs_test = imgs_test.astype(np.uint8)
# imgs_mask_train = imgs_mask_test.astype(np.uint8)



imgs_merged = imgs_mask_test + imgs_test

print(imgs_merged[10,11,100,:])

#imgs_merged = np.squeeze(imgs_merged, axis=4)
count_visualize = 1
count_processed = 0
pred_dir = 'test_merged'
if not os.path.exists(pred_dir):
    os.mkdir(pred_dir)
for x in range(0, imgs_merged.shape[0]):
    for y in range(0, imgs_merged.shape[1]):
        if (count_visualize > 1) and (count_visualize < 16):
            imsave(os.path.join(pred_dir, 'pred_' + str(count_processed) + '.png'), imgs_merged[x][y])
            count_processed += 1
        count_visualize += 1
        if count_visualize == 17:
            count_visualize = 1
        if (count_processed % 100) == 0:
            print('Done: {0}/{1} test images'.format(count_processed, imgs_merged.shape[0] * 14))





#count_visualize = 1
#count_processed = 0
#pred_dir = 'test_final'

#imgs_mask_test = np.squeeze(imgs_mask_test, axis=4)
#if not os.path.exists(pred_dir):
#    os.mkdir(pred_dir)
#for x in range(0, imgs_mask_test.shape[0]):
    # for y in range(0, imgs_mask_test.shape[1]):
    #     if (count_visualize > 1) and (count_visualize < 16):
    #         imsave(os.path.join(pred_dir, 'pred_' + str(count_processed) + '.png'), imgs_mask_test[x][y])
    #         count_processed += 1
    #     count_visualize += 1
    #     if count_visualize == 17:
    #         count_visualize = 1
    #     if (count_processed % 100) == 0:
    #         print('Done: {0}/{1} test images'.format(count_processed, imgs_mask_test.shape[0] * 14))




print(np.shape(imgs_merged))


# imgs_mask_train = np.around(imgs_mask_train, decimals=0)
# imgs_mask_train = imgs_mask_train.astype(np.uint8)
#
#


# print(imgs_mask_train[10,11,100,:,0])

# y = np.expand_dims(imgs_train, axis=4)
# print(np.shape(y))
# print(y[20,10,100,:,0])
# y = np.squeeze(y, axis=4)
# print(np.shape(y))
# print(y[20,10,100,:])

#print(imgs_train[30,10])

#print(imgs_train[0][10][10])

# y = np.expand_dims(imgs_train, axis=5)
# print(np.shape(y))
# print(y[30,10,:,:,0,0])

# count_processed = 0
# pred_dir = 'train_preprocessed'
# if not os.path.exists(pred_dir):
#     os.mkdir(pred_dir)
# for x in range(0, 500):
#     for y in range(0, imgs_train.shape[1]):
#         imsave(os.path.join(pred_dir, 'pre_processed_' + str(count_processed) + '.png'), imgs_train[x][y])
#         count_processed += 1
#         if (count_processed % 100) == 0:
#             print('Done: {0}/{1} train images'.format(count_processed, 500))
#
# count_processed = 0
# pred_dir = 'mask_preprocessed'
# if not os.path.exists(pred_dir):
#     os.mkdir(pred_dir)
# for x in range(0, 500):
#     for y in range(0, imgs_mask_train.shape[1]):
#         imsave(os.path.join(pred_dir, 'pre_processed_' + str(count_processed) + '.png'), imgs_mask_train[x][y])
#         count_processed += 1
#         if (count_processed % 100) == 0:
#             print('Done: {0}/{1} mask images'.format(count_processed, 500))
#
# count_processed = 0
# pred_dir = 'test_preprocessed'
# if not os.path.exists(pred_dir):
#     os.mkdir(pred_dir)
# for x in range(0, imgs_test.shape[0]):
#     for y in range(0, imgs_test.shape[1]):
#         imsave(os.path.join(pred_dir, 'pre_processed_' + str(count_processed) + '.png'), imgs_test[x][y])
#         count_processed += 1
#         if (count_processed % 100) == 0:
#             print('Done: {0}/{1} test images'.format(count_processed, 500))


