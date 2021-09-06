import csv
import numpy as np
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model,layers
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.applications.inception_v3 import InceptionV3
import os
from os import chdir,getcwd,path
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Conv2D, AveragePooling2D, Dense, Flatten

#path_inception=f"{getcwd()}/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"

#pre_trained_model=InceptionV3(
    #input_shape=(28,28,1),
    #include_top=False,
    #weights=None
#)

#local_weights_file=path_inception

#pre_trained_model.load_weights(local_weights_file)

#for layer in pre_trained_model.layers:
    #layer_trainable=False

#pre_trained_model.summary()

def get_data(filename):
    with open(filename) as training_file:
        reader=csv.reader(training_file)
        image=[]
        label1=[]

        next(reader,None)
        
        for row in reader:
            label=row[0]
            data=row[1:]
            img=np.array(data).reshape(28,28)

            image.append(img)
            label1.append(label)

        images=np.array(image).astype("float32")
        labels=np.array(label1).astype("float32")
    
    return images,labels

path_train_mnist_sign=f"{getcwd()}/sign_mnist_train.csv"
path_test_mnist_sign=f"{getcwd()}/sign_mnist_test.csv"

training_images,training_labels=get_data(path_train_mnist_sign)
testing_images,testing_labels=get_data(path_test_mnist_sign)

training_images=np.expand_dims(training_images,axis=3)
testing_images=np.expand_dims(testing_images,axis=3)


train_datagen=ImageDataGenerator(
    rescale=1/255,
    rotation_range=40,
    width_shift_range=.2,
    height_shift_range=.2,
    shear_range=.2,
    zoom_range=.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

training_generator=train_datagen.flow(
    training_images,
    training_labels,
    batch_size=128,
    
)
validation_datagen = ImageDataGenerator(
    rescale=1/255,
    rotation_range=40,
    width_shift_range=.2,
    height_shift_range=.2,
    shear_range=.2,
    zoom_range=.2,
    horizontal_flip=True,
    fill_mode="nearest"
)
validation_generator=validation_datagen.flow(training_images,training_labels,batch_size=128)

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

output_1=layers.Dense(26,activation=tf.nn.softmax,name="output_1")(x)

output_2=layers.Dense(26,activation=tf.nn.softmax,name="output_2")(x)

model=Model(inputs=[input_tensor_1],outputs=[output_1,output_2])


model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=["acc"])

#training_generator=np.array(training_generator).astype("float32")

model.fit(
    training_images,
    (training_labels,testing_labels),
    epochs=2,
)
