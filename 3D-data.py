from __future__ import print_function

import os
import glob
import numpy as np
from skimage.transform import resize
from skimage.io import imsave

from skimage.io import imread

data_path = './'

image_rows = int(400)
image_cols = int(400)
image_depth = 16

def create_train_data():
    train_data_path = os.path.join(data_path, 'train/')
    mask_data_path = os.path.join(data_path, 'masks/')
    dirs = os.listdir(train_data_path)
    total = int(len(dirs)*16*2)

    imgs = np.ndarray((total, image_depth, image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, image_depth, image_rows, image_cols), dtype=np.uint8)

    imgs_temp = np.ndarray((total, image_depth//2, image_rows, image_cols), dtype=np.uint8)
    imgs_mask_temp = np.ndarray((total, image_depth//2, image_rows, image_cols), dtype=np.uint8)

    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for dirr in sorted(os.listdir(train_data_path)):
        j = 0
        dirr = os.path.join(train_data_path, dirr)
        images = sorted(os.listdir(dirr))
        count = total
        for image_name in images:
            img = imread(os.path.join(dirr, image_name), as_grey=True)
            img = img.astype(np.uint8)
            img = np.array([img])
            imgs_temp[i,j] = img
            j += 1
            if j % (image_depth/2) == 0:
                j=0
                i += 1
                if (i % 100) == 0:
                    print('Done: {0}/{1} 3d images'.format(i, count))

    for x in range(0, imgs_temp.shape[0]-1):
        imgs[x]=np.append(imgs_temp[x], imgs_temp[x+1], axis=0)

    print('Loading of train data done.')

    i = 0
    for dirr in sorted(os.listdir(train_data_path)):
        j = 0
        dirr = os.path.join(mask_data_path, dirr)
        images = sorted(os.listdir(dirr))
        count = total
        for mask_name in images:
            img_mask = imread(os.path.join(dirr, mask_name), as_grey=True)
            img_mask = img_mask.astype(np.uint8)

            img_mask = np.array([img_mask])

            imgs_mask_temp[i,j] = img_mask

            j += 1
            if j % (image_depth/2) == 0:
                j = 0
                i += 1
                if (i % 100) == 0:
                    print('Done: {0}/{1} mask 3d images'.format(i, count))

    for x in range(0, imgs_mask_temp.shape[0]-1):
        imgs_mask[x]=np.append(imgs_mask_temp[x], imgs_mask_temp[x+1], axis=0)

    print('Loading of masks done.')


    imgs_mask = preprocess(imgs_mask)
    imgs = preprocess(imgs)

    print('Preprocessing of masks done.')

    np.save('imgs_train.npy', imgs)
    np.save('imgs_mask_train.npy', imgs_mask)

    imgs = preprocess_squeeze(imgs)
    imgs_mask = preprocess_squeeze(imgs_mask)

    count_processed = 0
    pred_dir = 'train_preprocessed'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for x in range(0, 30):
        for y in range(0, imgs.shape[1]):
            imsave(os.path.join(pred_dir, 'pre_processed_' + str(count_processed) + '.png'), imgs[x][y])
            count_processed += 1
            if (count_processed % 100) == 0:
                print('Done: {0}/{1} train images'.format(count_processed, 500))

    count_processed = 0
    pred_dir = 'mask_preprocessed'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for x in range(0, 30):
        for y in range(0, imgs_mask.shape[1]):
            imsave(os.path.join(pred_dir, 'pre_processed_' + str(count_processed) + '.png'), imgs_mask[x][y])
            count_processed += 1
            if (count_processed % 100) == 0:
                print('Done: {0}/{1} train images'.format(count_processed, 500))


    print('Saving to .npy files done.')



def load_train_data():
    imgs_train = np.load('imgs_train.npy')
    imgs_mask_train = np.load('imgs_mask_train.npy')
    return imgs_train, imgs_mask_train


def create_test_data():
    test_data_path = os.path.join(data_path, 'test/')
    dirs = os.listdir(test_data_path)
    total = int(len(dirs))*18

    imgs = np.ndarray((total, image_depth, image_rows, image_cols), dtype=np.uint8)

    i = 0
    j = 0
    print('-'*30)
    print('Creating test images...')
    print('-'*30)
    for dirr in sorted(os.listdir(test_data_path)):
        dirr = os.path.join(test_data_path, dirr)
        images = sorted(os.listdir(dirr))
        count = total
        for image_name in images:
            img = imread(os.path.join(dirr, image_name), as_grey=True)
            img = img.astype(np.uint8)

            img = np.array([img])

            imgs[i][j] = img

            j += 1

            # if j % (image_depth-2) == 0:
            #     imgs[0][i+1][0] = img

            if j % (image_depth-1) == 0:
                imgs[i+1][0] = img

            if j % image_depth == 0:
                imgs[i+1][1] = img
                j = 2
                i += 1
                if (i % 100) == 0:
                    print('Done: {0}/{1} test 3d images'.format(i, count))

    print('Loading done.')

    imgs = preprocess(imgs)

    np.save('imgs_test.npy', imgs)

    imgs = preprocess_squeeze(imgs)

    count_processed = 0
    pred_dir = 'test_preprocessed'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for x in range(0, imgs.shape[0]):
        for y in range(0, imgs.shape[1]):
            imsave(os.path.join(pred_dir, 'pre_processed_' + str(count_processed) + '.png'), imgs[x][y])
            count_processed += 1
            if (count_processed % 100) == 0:
                print('Done: {0}/{1} test images'.format(count_processed, imgs.shape[0]*imgs.shape[1]))

    print('Saving to .npy files done.')


def load_test_data():
    imgs_test = np.load('imgs_test.npy')
    return imgs_test


def preprocess(imgs):
    imgs = np.expand_dims(imgs, axis=4)
    print(' ---------------- preprocessed -----------------')
    return imgs

def preprocess_squeeze(imgs):
    imgs = np.squeeze(imgs, axis=4)
    print(' ---------------- preprocessed squeezed -----------------')
    return imgs


if __name__ == '__main__':
    create_train_data()
    create_test_data()
