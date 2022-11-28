'''
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/1/8 上午7:23
# @Author : yhlin
# @Site : 
# @File : data_process.py
# @Software: PyCharm
'''
from keras import Model
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import GlobalAveragePooling2D, Dense
from keras.optimizer_v2.adam import Adam
from keras.preprocessing.image import ImageDataGenerator

path = '../COVID-19_Radiography_Dataset'
classes = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]
num_classes = len(classes)
batch_size = 16
ze = 224

train_datagen = ImageDataGenerator(rescale=1./255,
                                  rotation_range=20,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2,
                                  horizontal_flip=True, validation_split=0.2)
test_datagen = ImageDataGenerator(rescale=1./255,
                                 validation_split=0.2)

train_generator = train_datagen.flow_from_directory(directory=path,
                                             target_size=(299, 299),
                                             class_mode='categorical',
                                             subset='training',
                                             shuffle=True, classes=classes,
                                             batch_size=batch_size,
                                             color_mode="grayscale")
test_generator = test_datagen.flow_from_directory(directory=path,
                                             target_size=(299, 299),
                                             class_mode='categorical',
                                             subset='validation',
                                             shuffle=False, classes=classes,
                                             batch_size=batch_size,
                                             color_mode="grayscale")


print(test_generator)

def mobile_net_v2(train_generator, test_generator, epochs, size):
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

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.5, min_lr=0.00001)

    early_stop = EarlyStopping(monitor='val_loss',
                               mode='min',
                               patience=5,
                               restore_best_weights=True)
    # model_checkpoint
    mc = ModelCheckpoint('mobilenet_v2.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    r = model.fit_generator(train_generator,
                  steps_per_epoch=len(train_generator) // batch_size,
                  validation_steps=len(train_generator) // batch_size,
                  validation_data=train_generator,
                  epochs=epochs,
                  verbose=2,
                  shuffle=True,
                            callbacks=[early_stop, mc])
    # model.evaluate_generator
    print("Train score:", model.evaluate_generator(train_generator))
    print("Test score:", model.evaluate_generator(test_generator))
    n_epochs = len(r.history['loss'])

    return r, model, n_epochs


if __name__ == '__main__':
    epochs = 1000
    r,model,n_epochs = mobile_net_v2(train_generator, test_generator,epochs,32)
    print(model.evaluate_generator(test_generator,verbose=True))
# plotLearningCurve(r,n_epochs)