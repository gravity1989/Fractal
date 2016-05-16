import numpy as np
import h5py
import os 

from readData2 import ReadHdf5, Convert3DTo2D, RemoveZeroLabelSlices
from readData2 import ZeroPadSlicesDivisibleByN

debug = 0
threshold = 0

img_array, label_array = ReadHdf5(debug)
image2Darray, label2Darray = Convert3DTo2D(img_array, label_array)
image2Darray, label2Darray = RemoveZeroLabelSlices(image2Darray, label2Darray, threshold=threshold)
image2Darray, label2Darray = ZeroPadSlicesDivisibleByN (image2Darray, label2Darray, number=16, debug=debug)

# U-net with keras functional API
from keras.models import Model









'''
# U - net 
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D
# from keras.layers import Merge

model = Sequential()
model.add( Convolution2D( 64, 3, 3, border_mode='same', input_shape=(1, 256, 256) ) )
model.add(Activation('relu'))
model.add( Convolution2D( 64, 3, 3) )
model.add(Activation('relu'))
model.add( MaxPooling2D (pool_size= (2,2), strides = (2,2)) )

model.add( Convolution2D( 128, 3, 3) )
model.add(Activation('relu'))
model.add( Convolution2D( 128, 3, 3) )
model.add(Activation('relu'))
model.add( MaxPooling2D (pool_size= (2,2), strides = (2,2)) )

model.add( Convolution2D( 256, 3, 3) )
model.add(Activation('relu'))
model.add( Convolution2D( 256, 3, 3) )
model.add(Activation('relu'))
model.add( MaxPooling2D (pool_size= (2,2), strides = (2,2)) )

model.add( Convolution2D( 512, 3, 3) )
model.add(Activation('relu'))
model.add( Convolution2D( 512, 3, 3) )
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add( MaxPooling2D (pool_size= (2,2), strides = (2,2)) )

model.add( Convolution2D( 1024, 3, 3) )
model.add(Activation('relu'))
model.add( Convolution2D( 1024, 3, 3) )
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Up- convolution to halve the input features 
model.add( UpSampling2D(size=(2,2)) )
model.add( Convolution2D( 512, 2, 2) )
# And some cropping of feature map from the contraction block



model.add(Flatten())



model.add(Dense(1024))

model.add(Dropout(0.5))


model.add(Dense(256))


model.add(Dense(5))
model.add(Activation('softmax'))


'''











