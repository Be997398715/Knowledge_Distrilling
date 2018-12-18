
# coding: utf-8
# In[1]:
import pickle
import numpy as np
import time
import sys  
sys.path.append('./models')
import matplotlib.pyplot as plt

import keras
from keras import backend as K
#from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.layers import Lambda, concatenate, Activation
from keras.losses import categorical_crossentropy as logloss
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from keras.models import Model
import matplotlib.pyplot as plt

from student_model import student, preprocess_input
from teacher_model import teacher_model


# In[2]:
# 开始下载数据集
t0 = time.time()  

DOWNLOAD = True
# CIFAR10 图片数据集
if(DOWNLOAD):
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()  # 32×32

else:
    pass

X_train = X_train.astype('float32')  # uint8-->float32
X_test = X_test.astype('float32')
X_train = preprocess_input(X_train)
X_test = preprocess_input(X_test)
print('训练样例:', X_train.shape, Y_train.shape,
      ', 测试样例:', X_test.shape, Y_test.shape)

n_classes = 10  # label为0~9共10个类别
# Convert class vectors to binary class matrices
Y_train = to_categorical(Y_train, n_classes)
Y_test = to_categorical(Y_test, n_classes)
print("取数据耗时: %.2f seconds ..." % (time.time() - t0))

use_teacher_model = False
if use_teacher_model:
    teacher_model = teacher_model(n_classes=n_classes)
    teacher_model.compile(
        optimizer=keras.optimizers.Adam(lr=1e-4), 
        loss='categorical_crossentropy', metrics=['accuracy', 'top_k_categorical_accuracy']
    )

    teacher_Y_train = teacher_model.predict(X_train, batch_size=512, verbose=1)
    Y_train = np.concatenate((Y_train, teacher_Y_train),axis=1)

    teacher_Y_test = teacher_model.predict(X_test, batch_size=512, verbose=1)
    Y_test = np.concatenate((Y_test, teacher_Y_test),axis=1)
# In[3]:
# define generators for training and validation data
train_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

val_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
train_datagen.fit(X_train)
val_datagen.fit(X_test)


temperature = 5.0

# In[4]: Model
model = student(weight_decay=1e-4, image_size=32, n_classes=n_classes)
# remove softmax
model.layers.pop()
# usual probabilities
logits = model.layers[-1].output
probabilities = Activation('softmax')(logits)
# softed probabilities
logits_T = Lambda(lambda x: x/temperature)(logits)
probabilities_T = Activation('softmax')(logits_T)
output = concatenate([probabilities, probabilities_T])
model = Model(model.input, output)
# now model outputs 20 dimensional vectors

model.load_weights('logs/weights/knowledge_distilling_weights_30_0.60.h5')

test = True
if test: 
    model = Model(model.input, probabilities)
    model.compile(
    optimizer=keras.optimizers.Adam(lr=1e-4), 
    loss='categorical_crossentropy', 
    metrics=['acc']
    )
train = False
if train:
    def softmax(x):
        return np.exp(x)/np.exp(x).sum()

    def knowledge_distillation_loss(y_true, y_pred, lambda_const):    
        y_true, logits = y_true[:, :10], y_true[:, 10:]
        # convert logits to soft targets
        y_soft = K.softmax(logits/temperature)
        y_pred, y_pred_soft = y_pred[:, :10], y_pred[:, 10:]    
        return lambda_const*logloss(y_true, y_pred) + logloss(y_soft, y_pred_soft)

    def accuracy(y_true, y_pred):
        y_true = y_true[:, :10]
        y_pred = y_pred[:, :10]
        return categorical_accuracy(y_true, y_pred)
    def top_5_accuracy(y_true, y_pred):
        y_true = y_true[:, :10]
        y_pred = y_pred[:, :10]
        return top_k_categorical_accuracy(y_true, y_pred)
    def categorical_crossentropy(y_true, y_pred):
        y_true = y_true[:, :10]
        y_pred = y_pred[:, :10]
        return logloss(y_true, y_pred)
    # logloss with only soft probabilities and targets
    def soft_logloss(y_true, y_pred):     
        logits = y_true[:, 10:]
        y_soft = K.softmax(logits/temperature)
        y_pred_soft = y_pred[:, 10:]    
        return logloss(y_soft, y_pred_soft)

    lambda_const = 0.2
    model.compile(
        optimizer=keras.optimizers.Adam(lr=1e-6), 
        loss=lambda y_true, y_pred: knowledge_distillation_loss(y_true, y_pred, lambda_const), 
        metrics=[accuracy, top_5_accuracy, categorical_crossentropy, soft_logloss]
    )

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=14, min_delta=0.01,verbose=1),
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=7, epsilon=0.007,verbose=1),
        ModelCheckpoint(monitor='val_accuracy',
                         filepath='logs/weights/knowledge_distilling_weights_{epoch:02d}_{val_accuracy:.2f}.h5',
                         save_best_only=True,
                         save_weights_only=True,
                         mode='auto',
                         verbose=1,
                         period=5)
                ]


    # In[5]: training
    batch_size = 32
    model.fit_generator(train_datagen.flow(X_train, Y_train, batch_size=batch_size),
                    steps_per_epoch=len(X_train)//batch_size, epochs=100,
                    validation_data=val_datagen.flow(X_test, Y_test, batch_size=batch_size), 
                    validation_steps=len(X_test)//batch_size,
                    callbacks=callbacks, initial_epoch=30, shuffle=True, verbose=2)


    # In[6]: Loss/epoch plots
    plt.plot(model.history.history['loss'], label='train')
    plt.plot(model.history.history['val_loss'], label='val')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('logloss')
    plt.show()

    plt.plot(model.history.history['accuracy'], label='train')
    plt.plot(model.history.history['val_accuracy'], label='val')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()


    plt.plot(model.history.history['top_k_categorical_accuracy'], label='train')
    plt.plot(model.history.history['val_top_k_categorical_accuracy'], label='val')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('top5_accuracy')
    plt.show()


# In[7]: Evaluate_Results
batch_size = 32
score = model.evaluate_generator(val_datagen.flow(X_test, Y_test), steps=len(X_test)/batch_size, use_multiprocessing=False, verbose=2)
print('loss:',score[0])
print('acc:',score[1])
