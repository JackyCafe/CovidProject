'''
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/1/10 下午8:24
# @Author : yhlin
# @Site : 
# @File : covid1.py
# @Software: PyCharm
'''
import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import seaborn as sns
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, BatchNormalization, MaxPooling2D, \
    GlobalAveragePooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical
import random
import cv2
import matplotlib.pyplot as plt
from imutils import paths
import itertools

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
print(gpus)


def generate_dataset_from_directory(folder_path, size=224, batch_size=32):
    image_generator = ImageDataGenerator(
        samplewise_center=True,  # Set each sample mean to 0.
        samplewise_std_normalization=True,  # Divide each input by its standard deviation]
        # rescale=1./255,
        validation_split=0.3
    )

    # create training and testing datasets
    train_data = image_generator.flow_from_directory(directory=
                                                     folder_path,
                                                     class_mode="categorical",
                                                     color_mode="rgb",
                                                     target_size=(size, size),
                                                     batch_size=batch_size,
                                                     shuffle=True,
                                                     seed=123,
                                                     subset="training"
                                                     )

    # create training and testing datasets
    val_data = image_generator.flow_from_directory(directory=
                                                   folder_path,
                                                   class_mode="categorical",
                                                   color_mode="rgb",
                                                   target_size=(size, size),
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   seed=123,
                                                   subset="validation"
                                                   )

    return train_data, val_data


folder_path = 'COVID-19_Radiography_Dataset/'
train_data, val_data = generate_dataset_from_directory(
    folder_path, size=128, batch_size=32)


def plotLearningCurve(history, epochs):
    epochRange = range(1, epochs + 1)
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].plot(epochRange, history.history['accuracy'], 'b', label='Training Accuracy')
    ax[0].plot(epochRange, history.history['val_accuracy'], 'r', label='Validation Accuracy')
    ax[0].set_title('Training and Validation accuracy')
    ax[0].set_xlabel('Epoch', fontsize=20)
    ax[0].set_ylabel('Accuracy', fontsize=20)
    ax[0].legend()
    ax[0].grid(color='gray', linestyle='--')
    ax[1].plot(epochRange, history.history['loss'], 'b', label='Training Loss')
    ax[1].plot(epochRange, history.history['val_loss'], 'r', label='Validation Loss')
    ax[1].set_title('Training and Validation loss')
    ax[1].set_xlabel('Epoch', fontsize=20)
    ax[1].set_ylabel('Loss', fontsize=20)
    ax[1].legend()
    ax[1].grid(color='gray', linestyle='--')
    plt.show()


def mobile_net_v2(train_data, test_data, epochs, size):
    # Building the models using Keras functional API
    print("----Building the models----")

    base_model = MobileNetV2(input_shape=(size, size, 3), include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    out = Dense(4, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=out)
    # models.summary()
    # Training the Convolutional Neural Network
    print("----Training the network----")
    model.compile(optimizer=Adam(0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    #     early_stop = EarlyStopping(monitor='val_loss',
    #                                mode='min',
    #                                patience=5,
    #                                restore_best_weights=True)
    # model_checkpoint
    mc = ModelCheckpoint('mobilenet_v2_224.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    r = model.fit(train_data,
                  validation_data=val_data,
                  epochs=epochs,
                  verbose=2,
                  batch_size=32,
                  callbacks=[mc])
    print("Train score:", model.evaluate(train_data))
    print("Test score:", model.evaluate(val_data))
    n_epochs = len(r.history['loss'])

    return r, model, n_epochs


epochs = 50
print(train_data.image_shape)
r, model, n_epochs = mobile_net_v2(train_data, val_data, epochs, 224)
print(model.evaluate(val_data, verbose=True))
plotLearningCurve(r, n_epochs)

# %%
#
# model = tf.keras.models.load_model('mobilenet_v2.h5')
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()
# open(f"mobilenet_v2.tflite", "wb").write(tflite_model)
#
#
# # %%
#
# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)
#
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')
#
#     print(cm)
#
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, cm[i, j],
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.savefig('confusion_matrix_v1.png')
#     plt.show()


# %%

# def score(model, shape):
#     predict = model.predict(X_test, batch_size=32)
#     predict = np.argmax(predict, axis=1)
#     cm = confusion_matrix(y_test.argmax(axis=1), predict)
#     cm_plot_labels = ['COVID', 'Opacity', 'Normal', 'Viral']
#     plot_confusion_matrix(cm, classes=cm_plot_labels, title='Confusion Matrix')
#     accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
#     recall = (cm[0][0]) / (cm[0][0] + cm[1][0])
#     presition = (cm[0][0]) / (cm[0][0] + cm[0][1])
#     specificity = (cm[1][1]) / (cm[1][1] + cm[0][1])
#     f1 = 2 * presition * recall / (presition + recall)
#     print(f'accuracy = {accuracy}\nrecall = {recall}\npresition = {presition}\nspecificity = {specificity}\nf1 = {f1}')
#     plt.title('Receiver Operating Characteristic')
#     plt.legend(loc='lower right')
#     plt.plot([0, 1], [0, 1], 'r--')
#     plt.xlim([0, 1])
#     plt.ylim([0, 1])
#     plt.ylabel('True Positive Rate')
#     plt.xlabel('False Positive Rate')
#     plt.savefig('roc.png')
#     plt.show()


# %%

def data_load():
    data = []
    labels = []
    output = './data/'
    dataset_path = 'COVID-19_Radiography_Dataset/'
    print("[INFO] loading images...")
    imagePaths = list(paths.list_images(dataset_path))

    pixel = 224
    arr = np.zeros(len(imagePaths), dtype=int)
    time = 0
    while time < len(imagePaths):
        i = random.randint(0, len(imagePaths) - 1)
        if arr[i] != 0:
            continue
        imagePath = imagePaths[i]

        label = imagePath.split(os.path.sep)[-2]

        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (pixel, pixel))

        data.append(image)
        labels.append(label)

        arr[i] = 1
        time += 1
    data = np.array(data)
    labels = np.array(labels)
    lb_encoder = LabelEncoder()
    labels = lb_encoder.fit_transform(labels)
    labels = to_categorical(labels)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)

    print('[INFO] finish loading image')
    return X_train, X_test, y_train, y_test, data, lb_encoder


def data_augmentation():
    generator = ImageDataGenerator()
    return generator

    # %%

    X_train, X_test, y_train, y_test, data, lb_encoder = data_load()
    np.save('x_train', X_train)
    np.save('X_test', X_test)
    np.save('y_train', y_train)
    np.save('y_test', y_test)
    generator = data_augmentation()

    score(model, 224)
