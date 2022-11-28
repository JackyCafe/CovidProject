'''
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/1/9 上午12:55
# @Author : yhlin
# @Site : 
# @File : data_process.py
# @Software: PyCharm
'''
import datetime
import logging

import self as self

import cv2
import os

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from imutils import paths
from sklearn.model_selection import train_test_split
import tensorflow as tf

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
	'[%(levelname)1.1s %(asctime)s %(module)s:%(lineno)d] %(message)s', datefmt='%Y%m%d %H:%M:%S')

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)

log_filename = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S.log")
fh = logging.FileHandler(log_filename)
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(ch)
logger.addHandler(fh)


class DataFeed:
    folder_path: str
    generator: ImageDataGenerator
    m_data = []
    m_labels = []
    size: int
    X_train: np.array
    y_train: np.array
    X_test: np.array
    y_test: np.array

    def __init__(self, folder_path: str, size=224):
        self.folder_path = folder_path

        self.generator = ImageDataGenerator()
        self.size = size

    def data_process(self):
        imagepaths = list(paths.list_images(self.folder_path))
        filenames = tf.constant(imagepaths)

        for imagePath in imagepaths:
            image = cv2.imread(imagePath, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (self.size, self.size))
            label = imagePath.split(os.path.sep)[-2]
            logging.info(imagePath)
            self.m_data.append(image)
            self.m_labels.append(label)
        labels = tf.constant(self.m_labels)
        print(filenames)
        print(self.m_labels)
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
        self.dataset = dataset.map(self._parse_function)

    # return self.data
    def _parse_function(self, filename, lb):
        image_string = tf.read_file(filename)
        # 讀取圖片
        image_decoded = tf.image.decode_image(image_string)
        # 調整大小
        image_resized = tf.image.resize_images(image_decoded, [self.size, self.size])
        return image_resized, lb


    def data_split(self, test_size=0.2):
        X_train, y_train, X_test, y_test = train_test_split(self.m_data, self.m_labels, test_size=test_size, stratify=self.labels, random_state=42)
        return X_train, y_train, X_test, y_test

    @property
    def data(self):
        return self.m_data

    @property
    def labels(self):
        return self.m_labels


if __name__ == '__main__':
    folder_path = './COVID-19_Radiography_Dataset'
    feed = DataFeed(folder_path, size=224)
    feed.data_process()
    print(feed.dataset)
    # X_train, y_train, X_test, y_test = feed.data_split(test_size=0.2)

