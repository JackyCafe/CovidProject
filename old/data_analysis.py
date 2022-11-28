'''
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/1/8 上午6:23
# @Author : yhlin
# @Site : 
# @File : data_analysis.py
# @Software: PyCharm
'''
from glob import glob

import numpy as np
import pandas as pd
import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image

np.random.seed(0)
tf.random.set_seed(0)

path = '../COVID-19_Radiography_Dataset'
diag_code_dict = {
    'COVID': 0,
    'Lung_Opacity': 1,
    'Normal': 2,
    'Viral Pneumonia': 3}

diag_title_dict = {
    'COVID': 'Covid-19',
    'Lung_Opacity': 'Lung Opacity',
    'Normal': 'Healthy',
    'Viral Pneumonia': 'Viral Pneumonia'}

#ave the file information to dict
#and save them into pandas
imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(path, '*','*.png'))}
covidData = pd.DataFrame.from_dict(imageid_path_dict, orient = 'index').reset_index()
covidData.columns = ['image_id','path']
classes = covidData.image_id.str.split('-').str[0]
covidData['diag'] = classes
covidData['target'] = covidData['diag'].map(diag_code_dict.get)
covidData['Class'] = covidData['diag'].map(diag_title_dict.get)
covidData.Class.unique()

covidData.isnull().sum()
# no null values
covidData.drop('diag', axis=1, inplace=True)

for i in covidData.Class.unique():
    count = covidData[covidData.Class == i].shape[0]
    print(f"{i} samples :", count, end=' ; ')
    print(f"{round(count/covidData.shape[0] * 100, 3)} % \n")

sample_imgs = []
classes = covidData.Class.unique()
for i in classes:
    for j in range(4):
        sample_imgs.append(cv2.imread(covidData[covidData.Class == i].iloc[j, 1]))

# plot each image from sample_imgs
plt.figure(figsize=(16, 16))

for i in range(0, 16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(sample_imgs[i])
    plt.axis('off')
    plt.title(classes[i // 4])
plt.show()

img = cv2.imread(covidData.iloc[0,1])
# shape of the image
print(img.shape)
covidData['image'] = covidData['path'].map(lambda x : np.array(Image.open(x).resize((75,75))))
covidData = covidData[['image_id', 'path', 'target', 'image', 'Class']] # rearranging the columns
mean_vals, max_vals, min_vals, std_vals = [], [], [], []

for i in range(covidData.shape[0]):
    mean_vals.append(covidData['image'][i].mean())
    max_vals.append(covidData['image'][i].max())
    min_vals.append(covidData['image'][i].min())
    std_vals.append(np.std(covidData['image'][i]))
    print(covidData)

raw_df = covidData.loc[:, ['image']]
raw_df['max'] = max_vals
raw_df['min'] = min_vals
raw_df['mean'] = mean_vals
raw_df['std'] = std_vals
raw_df['Class'] = covidData.loc[:, 'Class']
print(raw_df)
