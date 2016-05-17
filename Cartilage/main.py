import numpy as np
import h5py
import os 

from readData2 import ReadHdf5, Convert3DTo2D, RemoveZeroLabelSlices
from readData2 import ZeroPadSlicesDivisibleByN

debug = 1
threshold = 0

img_array, label_array = ReadHdf5()
image2Darray, label2Darray = Convert3DTo2D(img_array, label_array)
image2Darray, label2Darray = RemoveZeroLabelSlices(image2Darray, label2Darray, threshold=threshold, debug=debug)
image2Darray, label2Darray = ZeroPadSlicesDivisibleByN (image2Darray, label2Darray, number=16, debug=debug)

# U-net with keras functional API
from keras.models import Model
from keras.layers import Input, Dense, Activation, Dropout, merge, Flatten 
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D

sizeX = image2Darray.shape[1]
sizeY = image2Darray.shape[2]

batch_size = 10

input_img = Input(shape=(1, sizeX, sizeY)) # Confirm shape

conv_1a = Convolution2D(64, 3, 3, border_mode='same', activation='relu')(input_img)
conv_1b = Convolution2D(64, 3, 3, border_mode='same', activation='relu')(conv_1a)
max_pool_1 = MaxPooling2D (pool_size= (2,2), strides = (2,2)) (conv_1b)

conv_2a = Convolution2D(128, 3, 3, border_mode='same', activation='relu')(max_pool_1)
conv_2b = Convolution2D(128, 3, 3, border_mode='same', activation='relu')(conv_2a)
max_pool_2 = MaxPooling2D (pool_size= (2,2), strides = (2,2)) (conv_2b)

conv_3a = Convolution2D(256, 3, 3, border_mode='same', activation='relu')(max_pool_2)
conv_3b = Convolution2D(256, 3, 3, border_mode='same', activation='relu')(conv_3a)
max_pool_3 = MaxPooling2D (pool_size= (2,2), strides = (2,2)) (conv_3b)

conv_4a = Convolution2D(512, 3, 3, border_mode='same', activation='relu')(max_pool_3)
conv_4b = Convolution2D(512, 3, 3, border_mode='same', activation='relu')(conv_4a)
max_pool_4 = MaxPooling2D (pool_size= (2,2), strides = (2,2)) (conv_4b)
droput_1 = Dropout(0.5)(max_pool_4)

conv_5a = Convolution2D(1024, 3, 3, border_mode='same', activation='relu')(droput_1)
conv_5b = Convolution2D(1024, 3, 3, border_mode='same', activation='relu')(conv_5a)
droput_2 = Dropout(0.5)(conv_5b)

up_conv_4a = UpSampling2D(size=(2,2))(droput_2)
up_conv_4b = Convolution2D(512, 2, 2, border_mode='same', activation='relu')(up_conv_4a) 
merged_4 = merge([up_conv_4b, conv_4b], mode='concat')
up_conv_4c = Convolution2D(512, 3, 3, border_mode='same', activation='relu')(merged_4)
up_conv_4d = Convolution2D(512, 3, 3, border_mode='same', activation='relu')(up_conv_4c) 

up_conv_3a = UpSampling2D(size=(2,2))(up_conv_4d)
up_conv_3b = Convolution2D(256, 2, 2, border_mode='same', activation='relu')(up_conv_3a) 
merged_3 = merge([up_conv_3b, conv_3b], mode='concat')
up_conv_3c = Convolution2D(256, 3, 3, border_mode='same', activation='relu')(merged_3)
up_conv_3d = Convolution2D(256, 3, 3, border_mode='same', activation='relu')(up_conv_3c) 

up_conv_2a = UpSampling2D(size=(2,2))(up_conv_3d)
up_conv_2b = Convolution2D(128, 2, 2, border_mode='same', activation='relu')(up_conv_2a) 
merged_2 = merge([up_conv_2b, conv_2b], mode='concat')
up_conv_2c = Convolution2D(128, 3, 3, border_mode='same', activation='relu')(merged_2)
up_conv_2d = Convolution2D(128, 3, 3, border_mode='same', activation='relu')(up_conv_2c) 

up_conv_1a = UpSampling2D(size=(2,2))(up_conv_2d)
up_conv_1b = Convolution2D(64, 2, 2, border_mode='same', activation='relu')(up_conv_1a) 
merged_1 = merge([up_conv_1b, conv_1b], mode='concat')
up_conv_1c = Convolution2D(64, 3, 3, border_mode='same', activation='relu')(merged_1)
up_conv_1d = Convolution2D(64, 3, 3, border_mode='same', activation='relu')(up_conv_1c) 

up_conv_0 = Convolution2D(5, 1, 1, border_mode='same')(up_conv_1d) 
# up_conv_00 = Flatten()(up_conv_0)
# predictions = Dense(5, activation='softmax')(up_conv_00)






'''


model.add(Flatten())



model.add(Dense(1024))

model.add(Dropout(0.5))


model.add(Dense(256))


model.add(Dense(5))
model.add(Activation('softmax'))


'''











