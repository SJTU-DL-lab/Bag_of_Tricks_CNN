import keras
import numpy as np
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import add
from keras.layers import Activation
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
from keras.datasets import cifar10
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras.layers import BatchNormalization
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

'''
Tensorflow backend default
If you want to use th backend:
from keras import backend as K
'''

config = tf.ConfigProto()         #using GPU
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

STACK_NUM = 5
BASE_LR = 0.1
NUM_CLASSES = 10
WIDTH = 32
HEIGHT = 32
BATCH_SIZE = 256       #decrease the number if you run out your GPU memory
EPOCHES = 200
ITERATIONS = 50000//BATCH_SIZE +1
WEIGHT_DECAY = 1e-4           #according to the paper
FILE_PATH = "/cluster/home/it_stu25/dllab/model/best_model.h5"

def residual_block(x, filters, increase=False):
    stride = (1, 1)
    if increase:
        stride = (2, 2)

    o1 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(x))
    conv_1 = Conv2D(filters, kernel_size=(3, 3), strides=stride, padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(o1)
    o2 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(conv_1))
    conv_2 = Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(o2)
    if increase:
        projection = Conv2D(filters, kernel_size=(1, 1), strides=(2, 2), padding='same',
                            kernel_initializer="he_normal",
                            kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(o1)
        block = add([conv_2, projection])
    else:
        block = add([conv_2, x])
    return block


def residual_network(input_tensor, stack_num = 5):
    x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               kernel_initializer="he_normal",
               kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(input_tensor)

    for _ in range(stack_num):
        x = residual_block(x, 16, False)

    # input: 32x32x16 output: 16x16x32
    x = residual_block(x, 32, True)
    for _ in range(1, stack_num):
        x = residual_block(x, 32, False)

    # input: 16x16x32 output: 8x8x64
    x = residual_block(x, 64, True)
    for _ in range(1, stack_num):
        x = residual_block(x, 64, False)

    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    # input: 64 output: 10
    x = Dense(10, activation='softmax', kernel_initializer="he_normal",
              kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(x)
    return x

def color_preprocessing(x_train, x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        x_train[:, :, :, i] = (x_train[:, :, :, i] - mean[i]) / std[i]
        x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]
    return x_train, x_test

def scheduler(epoch):
    if epoch<5:
        return BASE_LR*(epoch+1)/5
    if epoch < 81:
        return 0.1
    if epoch < 122:
        return 0.01
    return 0.001


if __name__ == '__main__':

    (x_train, y_train),(x_test,y_test) = cifar10.load_data()
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    x_train, x_test = color_preprocessing(x_train, x_test)
    img_input = Input(shape=(HEIGHT,WIDTH,3))     #channel_last
    output = residual_network(img_input, STACK_NUM)
    resnet = Model(img_input, output)
    #other optimizers may achieve better performance
    sgd = optimizers.SGD(lr=0.1, momentum=0.9, nesterov=True)     #according the paper
    resnet.compile(loss='categorical_crossentropy',optimizer=sgd, metrics=['accuracy'])
    checkpoint = ModelCheckpoint(FILE_PATH, monitor='val_acc', verbose=1, save_best_only=True,mode='max')
    cbks = [LearningRateScheduler(schedule=scheduler),checkpoint]
    datagen = ImageDataGenerator(horizontal_flip=True,
                                 width_shift_range=0.125,
                                 height_shift_range=0.125,
                                 fill_mode='constant', cval=0.)
    datagen.fit(x_train)

    resnet.fit_generator(datagen.flow(x_train,y_train,batch_size=BATCH_SIZE),
                         steps_per_epoch=ITERATIONS,shuffle=1,
                         epochs=EPOCHES,
                         callbacks=cbks,
                         validation_data=(x_test,y_test))

    


