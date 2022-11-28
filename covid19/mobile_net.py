'''
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/1/10 上午12:07
# @Author : yhlin
# @Site : 
# @File : mobile_net.py
# @Software: PyCharm
'''
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Activation, Conv2D, Dense, Dropout, BatchNormalization, ReLU, \
    DepthwiseConv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Add
from tensorflow.keras.models import Model
from keras import regularizers
from datetime import datetime
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import pooling

from covid19.data_process import DataFeed

version = tf.__version__
# data
DATA_NUM_CLASSES = 4
DATA_CHANNELS = 3
DATA_ROWS = 32
DATA_COLS = 32
DATA_CROP_ROWS = 28
DATA_CROP_COLS = 28
# DATA_MEAN = np.array([[[125.30691805, 122.95039414, 113.86538318]]])  # CIFAR10
# DATA_STD_DEV = np.array([[[62.99321928, 62.08870764, 66.70489964]]])  # CIFAR10

# model
MODEL_LEVEL_0_BLOCKS = 4
MODEL_LEVEL_1_BLOCKS = 6
MODEL_LEVEL_2_BLOCKS = 3

# training
TRAINING_BATCH_SIZE = 32
TRAINING_SHUFFLE_BUFFER = 5000
TRAINING_BN_MOMENTUM = 0.99
TRAINING_BN_EPSILON = 0.001
TRAINING_LR_MAX = 0.001
# TRAINING_LR_SCALE        = 0.1
# TRAINING_LR_EPOCHS       = 2
TRAINING_LR_INIT_SCALE = 0.01
TRAINING_LR_INIT_EPOCHS = 5
TRAINING_LR_FINAL_SCALE = 0.01
TRAINING_LR_FINAL_EPOCHS = 55

# training (derived)
TRAINING_NUM_EPOCHS = TRAINING_LR_INIT_EPOCHS + TRAINING_LR_FINAL_EPOCHS
TRAINING_LR_INIT = TRAINING_LR_MAX * TRAINING_LR_INIT_SCALE
TRAINING_LR_FINAL = TRAINING_LR_MAX * TRAINING_LR_FINAL_SCALE
SAVE_MODEL_PATH = './save/model/'


# define the filter size
def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# define the calculation of each 'inverted Res_Block'
def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id):
    prefix = 'block_{}_'.format(block_id)

    in_channels = inputs.shape[-1]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs

    # Expand
    if block_id:
        x = Conv2D(expansion * in_channels, kernel_size=1, strides=1, padding='same', use_bias=False, activation=None,
                   kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5), name=prefix + 'expand')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'expand_BN')(x)
        x = ReLU(6., name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'

    # Depthwise
    x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None, use_bias=False, padding='same',
                        kernel_initializer="he_normal", depthwise_regularizer=regularizers.l2(4e-5),
                        name=prefix + 'depthwise')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'depthwise_BN')(x)
    x = ReLU(6., name=prefix + 'depthwise_relu')(x)

    # Project
    x = Conv2D(pointwise_filters, kernel_size=1, strides=1, padding='same', use_bias=False, activation=None,
               kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5), name=prefix + 'project')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'project_BN')(x)

    if in_channels == pointwise_filters and stride == 1:
        return Add(name=prefix + 'add')([inputs, x])
    return x


# Create Build
def create_model(rows, cols, channels, level_0_blocks, level_1_blocks, level_2_blocks, num_classes, lr_initial):
    # encoder - input
    alpha = 1.0
    include_top = True
    model_input = tf.keras.Input(shape=(rows, cols, channels), name='input_image')
    x = model_input

    first_block_filters = _make_divisible(32 * alpha, 8)

    # model architechture
    x = Conv2D(first_block_filters, kernel_size=3, strides=1, padding='same', use_bias=False,
               kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5), name='Conv1')(model_input)

    x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1, expansion=1, block_id=0)

    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1, expansion=6, block_id=1)
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1, expansion=6, block_id=2)

    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2, expansion=6, block_id=3)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=4)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=5)

    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=2, expansion=6, block_id=6)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=7)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=8)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=9)
    x = Dropout(rate=0.25)(x)

    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=10)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=11)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=12)
    x = Dropout(rate=0.25)(x)

    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, expansion=6, block_id=13)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, expansion=6, block_id=14)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, expansion=6, block_id=15)
    x = Dropout(rate=0.25)(x)

    x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1, expansion=6, block_id=16)
    x = Dropout(rate=0.25)(x)

    # define filter size (last block)
    if alpha > 1.0:
        last_block_filters = _make_divisible(1280 * alpha, 8)
    else:
        last_block_filters = 1280

    x = Conv2D(last_block_filters, kernel_size=1, use_bias=False, kernel_initializer="he_normal",
               kernel_regularizer=regularizers.l2(4e-5), name='Conv_1')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv_1_bn')(x)
    x = ReLU(6., name='out_relu')(x)

    if include_top:
        x = GlobalAveragePooling2D(name='global_average_pool')(x)
        x = Dense(DATA_NUM_CLASSES, activation='softmax', use_bias=True, name='Logits')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # create model of MobileNetV2 (for CIFAR-10)
    model = Model(inputs=model_input, outputs=x, name='mobilenetv2_covid10')

    model.compile(optimizer=tf.keras.optimizers.Adam(lr_initial), loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def lr_schedule(epoch):
    # staircase

    if epoch < TRAINING_LR_INIT_EPOCHS:
        lr = (TRAINING_LR_MAX - TRAINING_LR_INIT)*(float(epoch)/TRAINING_LR_INIT_EPOCHS) + TRAINING_LR_INIT
    else:
        lr = (TRAINING_LR_MAX - TRAINING_LR_FINAL)

    return lr

# plot training accuracy and loss curves
def plot_training_curves(history):

    # training and validation data accuracy
    acc     = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    # training and validation data loss
    loss     = history.history['loss']
    val_loss = history.history['val_loss']

    # plot accuracy
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    # plot loss
    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 2.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

if __name__ == '__main__':

    callbacks = [tf.keras.callbacks.LearningRateScheduler(lr_schedule),
                 tf.keras.callbacks.ModelCheckpoint(filepath=SAVE_MODEL_PATH + 'model_{epoch}.h5', save_best_only=True, monitor='val_loss', verbose=1)]

    folder_path = './COVID-19_Radiography_Dataset'
    feed = DataFeed(folder_path, size=224)
    feed.data_process()
    X_train, y_train, X_test, y_test = feed.data_split(test_size=0.2)
    model_name      = 'mobilenet_v2-like__' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # build model
    # create and compile model
    model = create_model(DATA_CROP_ROWS, DATA_CROP_COLS, DATA_CHANNELS,
                         MODEL_LEVEL_0_BLOCKS, MODEL_LEVEL_1_BLOCKS, MODEL_LEVEL_2_BLOCKS,
                         DATA_NUM_CLASSES, TRAINING_LR_MAX)
    model.summary()
    tf.keras.utils.plot_model(model, 'covid-19.png', show_shapes=True)

    initial_epoch_num = 0

    generator = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)
    epochs = 100
    early_stop = EarlyStopping(monitor='val_loss',
                               mode='min',
                               patience=5,
                               restore_best_weights=True)
    # model_checkpoint
    mc = ModelCheckpoint('models/mobilenet_v2_1.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

    history = model.fit_generator(generator.flow(X_train, y_train, batch_size=TRAINING_BATCH_SIZE),
                                  validation_data=(X_test, y_test),
                                  epochs=epochs,
                                  verbose=2,
                                  validation_steps=len(X_test),
                                  callbacks=[early_stop, mc],
                                  )



    # history = model.fit_generator(x=X_train,
    #                 epochs=60,
    #                 verbose=1,
    #                 callbacks=callbacks,
    #                 validation_data=X_test,
    #                 initial_epoch=initial_epoch_num)

    # history = model.fit(x=X_train,
    #                     epochs=60,
    #                     verbose=1,
    #                     callbacks=callbacks,
    #                     validation_data=X_test,
    #                     initial_epoch=initial_epoch_num)
    plot_training_curves(history)
# test

    test_loss, test_accuracy = model.evaluate(x=X_test)
    print('Test loss:     ', test_loss)
    print('Test accuracy: ', test_accuracy)