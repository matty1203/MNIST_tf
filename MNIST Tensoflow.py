# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 00:50:08 2021

@author: Mathews
"""

import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds

mnist_data,mnist_info=tfds.load(name="mnist",as_supervised=True,with_info=True)

mnist_train,mnist_test=mnist_data['train'],mnist_data['test']

num_validation=0.1*mnist_info.splits['train'].num_examples

num_validation=tf.cast(num_validation,tf.int64)


num_test=0.1*mnist_info.splits['test'].num_examples

num_test=tf.cast(num_test,tf.int64)

def imageScaler(image,label):
    scaled=tf.cast(image,tf.float32)/255.0
    return image,label

scaled_train_validation_data=mnist_train.map(imageScaler)
test_data=mnist_test.map(imageScaler)

BUFFER_SIZE=10000
shuffled_train_and_validation_data=scaled_train_validation_data.shuffle(BUFFER_SIZE)

validation_data=shuffled_train_and_validation_data.take(num_validation)
train_data=shuffled_train_and_validation_data.skip(num_validation)

BATCH_SIZE=1000

train_data=train_data.batch(BATCH_SIZE)
validation_data=validation_data.batch(num_validation)
test_data=test_data.batch(num_test)

validation_inputs,validation_targets=next(iter(validation_data))

INPUT_SIZE=28*28
TARGET_SIZE=10
HIDDEN_LAYER_SIZE=200
model=tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28,28,1)),
                           
                           #tf.keras.layers.Dense() -> it is actually adding a hidden layer
                           tf.keras.layers.Dense(HIDDEN_LAYER_SIZE,activation='relu'),
                           tf.keras.layers.Dense(HIDDEN_LAYER_SIZE,activation='relu'),
                           tf.keras.layers.Dense(HIDDEN_LAYER_SIZE,activation='relu'),
                           tf.keras.layers.Dense(HIDDEN_LAYER_SIZE,activation='sigmoid'),
                           tf.keras.layers.Dense(TARGET_SIZE,activation='softmax')
                         
                           ])
custom_optimizer=tf.keras.optimizers.Adam(learning_rate=0.002)
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
NUM_EPOCHS=10
model.fit(train_data,epochs=NUM_EPOCHS,validation_data=(validation_inputs,validation_targets),verbose=2)


test_loss,test_acc=model.evaluate(test_data)