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
image_depth = 1

def create_train_data():
    train_data_path = os.path.join(data_path, 'train/')
    mask_data_path = os.path.join(data_path, 'masks/')
    dirs = os.listdir(train_data_path)

    total = 0
    for root, dirs, files in os.walk(train_data_path):
        total += len(files)


    imgs_temp = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_mask_temp = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)

    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for dirr in sorted(os.listdir(train_data_path)):
        dirr = os.path.join(train_data_path, dirr)
        images = sorted(os.listdir(dirr))
        count = total
        for image_name in images:
            img = imread(os.path.join(dirr, image_name), as_grey=True)
            img = img.astype(np.uint8)
            img = np.array([img])
            imgs_temp[i] = img
            i += 1
            if (i % 100) == 0:
                print('Done: {0}/{1} 2d images'.format(i, count))

    imgs = imgs_temp

    print('Loading of train data done.')

    i = 0
    for dirr in sorted(os.listdir(train_data_path)):
        dirr = os.path.join(mask_data_path, dirr)
        images = sorted(os.listdir(dirr))
        count = total
        for mask_name in images:
            img_mask = imread(os.path.join(dirr, mask_name), as_grey=True)
            img_mask = img_mask.astype(np.uint8)
            img_mask = np.array([img_mask])
            imgs_mask_temp[i] = img_mask
            i += 1
            if (i % 100) == 0:
                print('Done: {0}/{1} mask 2d images'.format(i, count))

    imgs_mask = imgs_mask_temp

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
    for x in range(0, 256):
        imsave(os.path.join(pred_dir, 'pre_processed_' + str(count_processed) + '.png'), imgs[x])
        count_processed += 1
        if (count_processed % 100) == 0:
            print('Done: {0}/{1} train images'.format(count_processed, 500))

    count_processed = 0
    pred_dir = 'mask_preprocessed'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for x in range(0, 256):
        imsave(os.path.join(pred_dir, 'pre_processed_' + str(count_processed) + '.png'), imgs_mask[x])
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
    total = 0
    for root, dirs, files in os.walk(test_data_path):
        total += len(files)

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)

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
            imgs[i] = img
            i += 1
            if (i % 100) == 0:
                print('Done: {0}/{1} test 2d images'.format(i, count))

    print('Loading done.')

    imgs = preprocess(imgs)

    np.save('imgs_test.npy', imgs)

    imgs = preprocess_squeeze(imgs)

    count_processed = 0
    pred_dir = 'test_preprocessed'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for x in range(0, imgs.shape[0]):
        imsave(os.path.join(pred_dir, 'pre_processed_' + str(count_processed) + '.png'), imgs[x])
        count_processed += 1
        if (count_processed % 100) == 0:
            print('Done: {0}/{1} test images'.format(count_processed, imgs.shape[0]))

    print('Saving to .npy files done.')


def load_test_data():
    imgs_test = np.load('imgs_test.npy')
    return imgs_test


def preprocess(imgs):
    imgs = np.expand_dims(imgs, axis=3)
    print(' ---------------- preprocessed -----------------')
    return imgs

def preprocess_squeeze(imgs):
    imgs = np.squeeze(imgs, axis=3)
    print(' ---------------- preprocessed squeezed -----------------')
    return imgs


if __name__ == '__main__':
    create_train_data()
    create_test_data()
