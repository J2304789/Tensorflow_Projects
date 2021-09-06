import csv
import os
from os import chdir,getcwd,path
import numpy as np
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Concatenate

def get_data(filename):
    with open(filename) as training_file:
        reader=csv.reader(training_file)
        imgs=[]
        labels=[]

        next(reader,None)

        for row in reader:
            label=row[0]
            data=row[1:]
            img=np.array(data).reshape(28,28)
            imgs.append(img)
            labels.append(label)

        images=np.array(imgs).astype("float")
        labels=np.array(labels).astype("float")

    return (images,labels)

path_sign_mnist_train=f"{getcwd()}sign_mnist_train.csv"
path_sign_mnist_test=f"{getcwd()}sign_mnist_test.csv"

training_images,training_labels=get_data(path_sign_mnist_train)
testing_images,testing_labels=get_data(path_sign_mnist_train)

#print(training_images.shape)
#print(training_labels.shape)
#print(testing_images.shape)
#print(testing_labels.shape)

training_images=np.array(training_images)
testing_images=np.array(testing_images)
training_labels=np.array(training_labels)
testing_labels=np.array(testing_labels)
class myCallback(tf.keras.callbacks.Callback):
    def end(self,epoch,logs={}):
        if logs.get("acc")>.9:
            print("acc reached")
            self.model.stop_training=True
        else:
            print("acc not reached")

training_images=np.expand_dims(training_images,axis=3)
testing_images=np.expand_dims(testing_images,axis=3)
#training_images=training_images/255
#testing_images=testing_images/255
train_datagen=ImageDataGenerator(
    rescale=1/255,
    #rotation_range=40,
    #width_shift_range=.2,
    #height_shift_range=.2,
    #shear_range=.2,
    #zoom_range=.2,
    #horizontal_flip=True,
    #fill_mode="nearest"
)

training_generator=train_datagen.flow(
    training_images,
    training_labels,
    batch_size=128,
    
)
validation_datagen = ImageDataGenerator(
    rescale=1/255,
    #rotation_range=40,
    #width_shift_range=.2,
    #height_shift_range=.2,
    #shear_range=.2,
    #zoom_range=.2,
    #horizontal_flip=True,
    #fill_mode="nearest"
)
validation_generator=validation_datagen.flow(
    training_images,
    training_labels,
    batch_size=128)

#Sequential API
#model = tf.keras.models.Sequential([
    #tf.keras.layers.Conv2D(32,(3,3),activation=tf.nn.relu,input_shape=[28,28,1]),
    #tf.keras.layers.MaxPool2D(2,2),
    #tf.keras.layers.Conv2D(64,(3,3),activation=tf.nn.relu,input_shape=[28,28,1]),
    #tf.keras.layers.MaxPool2D(2,2),
    #tf.keras.layers.Flatten(),
    #tf.keras.layers.Dropout(.2),
    #tf.keras.layers.Dense(128,activation=tf.nn.relu),
    #tf.keras.layers.Dense(26,activation=tf.nn.softmax)
    
#])

#Functional API
input_tensor_1=layers.Input(shape=(28,28,1))
x=layers.Conv2D(16,(3,3),activation=tf.nn.relu,input_shape=[28,28,1])(input_tensor_1)
x=layers.MaxPool2D(2,2)(x)
x=layers.Conv2D(16,(3,3),activation=tf.nn.relu,input_shape=[28,28,1])(x)
x=layers.MaxPool2D(2,2)(x)
x=layers.Conv2D(16,(3,3),activation=tf.nn.relu,input_shape=[28,28,1])(x)
x=layers.MaxPool2D(2,2)(x)
x=layers.Flatten()(x)
x=layers.Dropout(.2)(x)
x=layers.Dense(1024,activation=tf.nn.relu)(x)
x=layers.Dense(26,activation=tf.nn.softmax)(x)

input_tensor_2=layers.Input(shape=(28,28,1))
y=layers.Conv2D(16,(3,3),activation=tf.nn.relu,input_shape=[28,28,1])(input_tensor_2)
y=layers.MaxPool2D(2,2)(y)
y=layers.Conv2D(16,(3,3),activation=tf.nn.relu,input_shape=[28,28,1])(y)
y=layers.MaxPool2D(2,2)(y)
y=layers.Conv2D(16,(3,3),activation=tf.nn.relu,input_shape=[28,28,1])(y)
y=layers.MaxPool2D(2,2)(y)
y=layers.Flatten()(y)
y=layers.Dropout(.2)(y)
y=layers.Dense(1024,activation=tf.nn.relu)(y)
y=layers.Dense(26,activation=tf.nn.softmax)(y)

concat=keras.layers.concatenate([x,y])
output=layers.Dense(26,name="output")(concat)
model=keras.Model(inputs=[input_tensor_1,input_tensor_2],outputs=[output])

model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=["acc"])

callbacks=myCallback()

training_generator = np.array(training_generator).astype(dtype="uint8")

validation_generator = np.array(validation_generator).astype(dtype="uint8")

history=model.fit(
    #(training_images,testing_images),
    (training_generator,validation_generator),
    #training_labels,
    epochs=2
)
#history=model.fit(
    #(training_generator,validation_generator),
    #epochs=2
#)

