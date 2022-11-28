import itertools
import os
import random
import shutil
from distutils.file_util import copy_file

import cv2
import matplotlib.pyplot as plt
from imutils import paths

from keras import Model
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import GlobalAveragePooling2D, Dense, MaxPooling2D, Flatten, Dropout
from keras.optimizer_v2.adam import Adam
from keras.utils.np_utils import to_categorical
from keras_preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

SIZE = 224
BATCH_SIZE = 32

def data_load():
    data = []
    labels = []
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
        # extract the class label from the filename
        label = imagePath.split(os.path.sep)[-2]
        # load the image, swap color channels, and resize it to be a fixed
        # 224x224 pixels while ignoring aspect ratio
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (pixel, pixel))
        # update the data and labels lists, respectively
        data.append(image)
        labels.append(label)
        #
        arr[i] = 1
        time += 1
    # convert the data and labels to NumPy arrays while scaling the pixel
    # intensities to the range [0, 1]
    data = np.array(data)
    labels = np.array(labels)
    lb_encoder = LabelEncoder()
    labels = lb_encoder.fit_transform(labels)
    labels = to_categorical(labels)

    # Split the data into training and testing using the 80% of training and 20% to testing
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)


    # print(data)
    print('[INFO] finish loading image')
    return X_train, X_test, y_train, y_test, data, lb_encoder


def data_augmentation():
    generator = ImageDataGenerator(rotation_range=15, fill_mode='nearest')
    return generator


def mobile_net_v2(generator, X_train, X_test, y_train, y_test, size, epochs, lb_encoder):
    # Building the models using Keras functional API
    print("----Building the models----")

    base_model = MobileNetV2(input_shape=(size, size, 3), weights='imagenet', include_top=False)
    # 鎖權重
    # for layers in base_model.layers:
    #     layers.trainable = False

    x = base_model.output
    x = MaxPooling2D(pool_size=(4, 4))(x)
    x = Flatten(name='Flatten')(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(64, activation='relu')(x)
    out = Dense(4, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=out)
    summary = model.summary()
    print(summary)

    print("----Training the network----")
    model.compile(optimizer=Adam(0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss',
                               mode='min',
                               patience=5,
                               restore_best_weights=True)
    # model_checkpoint
    mc = ModelCheckpoint('models/mobilenet_v2.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

    with tf.device('/gpu:0'):
        print('[INFO] Begining training...')
        history = model.fit_generator(generator.flow(X_train, y_train, batch_size=BATCH_SIZE),
                                      validation_data=(X_test, y_test),
                                      epochs=epochs,
                                      verbose=2,
                                      validation_steps= len(X_test),
                                      callbacks=[early_stop, mc],
                                      )


        print('[INFO] Saving model...')
        model.save('models/mobilenet_v2.h5')
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        # Save the model.
        with open("tflite/mobilenetv2.tflite", 'wb') as f:
            f.write(tflite_model)

        print("Train score:", model.evaluate(X_train,y_train))
        print("Test score:", model.evaluate(X_test,y_test))
        n_epochs = len(history.history['loss'])
        print("[INFO] evaluating network...")
        predIdxs = model.predict(X_test, batch_size=BATCH_SIZE)
        # for each image in the testing set we need to find the index of the
        # label with corresponding largest predicted probability
        predIdxs = np.argmax(predIdxs, axis=1)
        # show a nicely formatted classification report
        # lb_encoder = LabelEncoder()
        print(classification_report(y_test.argmax(axis=1), predIdxs, target_names=lb_encoder.classes_))

        #confusion_matrix
        cm = confusion_matrix(y_test.argmax(axis=1), predIdxs)
        total = sum(sum(cm))
        acc = (cm[0, 0] + cm[1, 1]) / total
        sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
        # show the accuracy, sensitivity, and specificity of the test
        print("accuracy: {:.4f}".format(acc))
        print("sensitivity: {:.4f}".format(sensitivity))
        print("specificity: {:.4f}".format(specificity))
        return history, model, n_epochs,cm


def plotLearningCurve(history, epochs):
    epochRange = range(1, epochs + 1)
    fig, ax = plt.subplots(2, 1, figsize=(20, 10))
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
    plt.savefig('score.png')
    plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')
    plt.show()


def score(model,shape):
      predict = model.predict(X_test, batch_size=BATCH_SIZE)
      predict = np.argmax(predict, axis=1)
      cm = confusion_matrix(y_test.argmax(axis=1), predict)
      cm_plot_labels = ['COVID','Opacity','Normal','Viral']
      plot_confusion_matrix(cm, classes=cm_plot_labels, title='Confusion Matrix')
      accuracy = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
      recall = (cm[0][0])/(cm[0][0]+cm[1][0])
      presition = (cm[0][0])/(cm[0][0]+cm[0][1])
      specificity = (cm[1][1])/(cm[1][1]+cm[0][1])
      f1 = 2*presition*recall/(presition+recall)
      print(f'accuracy = {accuracy}\nrecall = {recall}\npresition = {presition}\nspecificity = {specificity}\nf1 = {f1}')
      plt.title('Receiver Operating Characteristic')
      plt.legend(loc = 'lower right')
      plt.plot([0, 1], [0, 1],'r--')
      plt.xlim([0, 1])
      plt.ylim([0, 1])
      plt.ylabel('True Positive Rate')
      plt.xlabel('False Positive Rate')
      plt.savefig('roc.png')
      plt.show()

if __name__ == '__main__':
    X_train, X_test, y_train, y_test, data, lb_encoder = data_load()
    generator = data_augmentation()
    epochs = 100
    history, model, n_epochs,cm= mobile_net_v2(generator, X_train, X_test, y_train, y_test, 224, epochs,lb_encoder)
    plotLearningCurve(history, n_epochs)
    model_dir = 'models/mobilenet_v2.h5'
    model = tf.keras.models.load_model(model_dir)

    score(model,224)
    # data_labels()
    # epochs = 10
    # batch_size= BATCH_SIZE
    # train_data,vaild_data,test_data=data_augmentation()
    # model_dir = 'models\\mobilenet_v2_1.h5'
    # model = tf.keras.models.load_model(model_dir)
    # # model.evaluate(test_data)
    #
    # # print(len(test_data))
    # # r,model,n_epochs = mobile_net_v2(train_data,vaild_data,test_data,1000,SIZE)
    # # plotLearningCurve(r, n_epochs)