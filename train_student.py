
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
import matplotlib.pyplot as plt

from student_model import student, preprocess_input



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

nb_classes = 10  # label为0~9共10个类别
# Convert class vectors to binary class matrices
Y_train = to_categorical(Y_train, nb_classes)
Y_test = to_categorical(Y_test, nb_classes)
print("取数据耗时: %.2f seconds ..." % (time.time() - t0))



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



# In[4]: Model
model = student(weight_decay=1e-4, image_size=32, n_classes=10)
print(model.count_params())

model.compile(
    optimizer=keras.optimizers.Adam(lr=1e-4), 
    loss='categorical_crossentropy', metrics=['accuracy', 'top_k_categorical_accuracy']
)

callbacks = [
    EarlyStopping(monitor='val_acc', patience=10, min_delta=0.01,verbose=1),
    ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=3, epsilon=0.007,verbose=1),
    ModelCheckpoint(monitor='val_acc',
                     filepath='logs/weights/student_weights_{epoch:02d}_{val_acc:.2f}.h5',
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
                callbacks=callbacks, initial_epoch=0, shuffle=True, verbose=2)


# In[6]: Loss/epoch plots
plt.plot(model.history.history['loss'], label='train')
plt.plot(model.history.history['val_loss'], label='val')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('logloss')
plt.show()

plt.plot(model.history.history['acc'], label='train')
plt.plot(model.history.history['val_acc'], label='val')
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
score = model.evaluate_generator(val_datagen.flow(X_test, Y_test), steps=len(X_test)/batch_size, use_multiprocessing=False, verbose=2)
print('loss:',score[0])
print('acc:',score[1])
