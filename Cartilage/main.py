import numpy as np
import h5py
import os 

from readData2 import ReadHdf5, Convert3DTo2D, RemoveZeroLabelSlices
from readData2 import ZeroPadSlicesDivisibleByN

root_dir = '/Users/A/Documents/Fractal/Cartilage/TrainingData-A/'
debug = 1
threshold = 0
noImages = 3

img_array, label_array = ReadHdf5(root_dir, noImages)
image2Darray, label2Darray = Convert3DTo2D(img_array, label_array)
image2Darray, label2Darray = RemoveZeroLabelSlices(image2Darray, label2Darray, threshold=threshold, debug=debug)
image2Darray, label2Darray = ZeroPadSlicesDivisibleByN (image2Darray, label2Darray, number=16, debug=debug)

# U-net with keras functional API
from keras.models import Model
from keras.layers import Input, Dense, Activation, Dropout, merge, Flatten, Reshape
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D

sizeX = image2Darray.shape[1]
sizeY = image2Darray.shape[2]

batch_size = 10

input_img = Input(shape=(1, sizeX, sizeY)) # Confirm shape
if debug:
    print ("Size of input image is: ", sizeX, " ", sizeY)

conv_1a = Convolution2D(64, 3, 3, border_mode='same', activation='relu')(input_img)
conv_1b = Convolution2D(64, 3, 3, border_mode='same', activation='relu')(conv_1a)
max_pool_1 = MaxPooling2D (pool_size= (2,2), strides = (2,2)) (conv_1b)
if debug: 
  print("max_pool_1 ", Model(input=input_img, output=max_pool_1).output_shape )

conv_2a = Convolution2D(128, 3, 3, border_mode='same', activation='relu')(max_pool_1)
conv_2b = Convolution2D(128, 3, 3, border_mode='same', activation='relu')(conv_2a)
max_pool_2 = MaxPooling2D (pool_size= (2,2), strides = (2,2)) (conv_2b)
if debug:
  print("max_pool_2 ", Model(input=input_img, output=max_pool_2).output_shape )

conv_3a = Convolution2D(256, 3, 3, border_mode='same', activation='relu')(max_pool_2)
conv_3b = Convolution2D(256, 3, 3, border_mode='same', activation='relu')(conv_3a)
max_pool_3 = MaxPooling2D (pool_size= (2,2), strides = (2,2)) (conv_3b)
if debug:
  print("max_pool_3 ", Model(input=input_img, output=max_pool_3).output_shape )

conv_4a = Convolution2D(512, 3, 3, border_mode='same', activation='relu')(max_pool_3)
conv_4b = Convolution2D(512, 3, 3, border_mode='same', activation='relu')(conv_4a)
max_pool_4 = MaxPooling2D (pool_size= (2,2), strides = (2,2)) (conv_4b)
droput_1 = Dropout(0.5)(max_pool_4)
if debug:
  print("conv_4b ", Model(input=input_img, output=conv_4b).output_shape )
  print("droput_1 ", Model(input=input_img, output=droput_1).output_shape )

conv_5a = Convolution2D(1024, 3, 3, border_mode='same', activation='relu')(droput_1)
conv_5b = Convolution2D(1024, 3, 3, border_mode='same', activation='relu')(conv_5a)
droput_2 = Dropout(0.5)(conv_5b)
if debug:
  print("droput_2 ", Model(input=input_img, output=droput_2).output_shape )

up_conv_4a = UpSampling2D(size=(2,2))(droput_2)
up_conv_4b = Convolution2D(512, 2, 2, border_mode='same', activation='relu')(up_conv_4a) 
merged_4 = merge([up_conv_4b, conv_4b], mode='concat', concat_axis=1) # 1 adds along 512 - 512 : gives 1024
up_conv_4c = Convolution2D(512, 3, 3, border_mode='same', activation='relu')(merged_4)
up_conv_4d = Convolution2D(512, 3, 3, border_mode='same', activation='relu')(up_conv_4c) 
if debug:
  print("up_conv_4a ", Model(input=input_img, output=up_conv_4a).output_shape )
  print("up_conv_4b ", Model(input=input_img, output=up_conv_4b).output_shape )
  print("merged_4 ", Model(input=input_img, output=merged_4).output_shape )
  print("up_conv_4c ", Model(input=input_img, output=up_conv_4c).output_shape )
  print("up_conv_4d ", Model(input=input_img, output=up_conv_4d).output_shape )

up_conv_3a = UpSampling2D(size=(2,2))(up_conv_4d)
up_conv_3b = Convolution2D(256, 2, 2, border_mode='same', activation='relu')(up_conv_3a) 
merged_3 = merge([up_conv_3b, conv_3b], mode='concat', concat_axis=1)
up_conv_3c = Convolution2D(256, 3, 3, border_mode='same', activation='relu')(merged_3)
up_conv_3d = Convolution2D(256, 3, 3, border_mode='same', activation='relu')(up_conv_3c) 
if debug:
  print("up_conv_3d ", Model(input=input_img, output=up_conv_3d).output_shape )

up_conv_2a = UpSampling2D(size=(2,2))(up_conv_3d)
up_conv_2b = Convolution2D(128, 2, 2, border_mode='same', activation='relu')(up_conv_2a) 
merged_2 = merge([up_conv_2b, conv_2b], mode='concat', concat_axis=1)
up_conv_2c = Convolution2D(128, 3, 3, border_mode='same', activation='relu')(merged_2)
up_conv_2d = Convolution2D(128, 3, 3, border_mode='same', activation='relu')(up_conv_2c) 
if debug:
  print("up_conv_2d ", Model(input=input_img, output=up_conv_2d).output_shape )

up_conv_1a = UpSampling2D(size=(2,2))(up_conv_2d)
up_conv_1b = Convolution2D(64, 2, 2, border_mode='same', activation='relu')(up_conv_1a) 
merged_1 = merge([up_conv_1b, conv_1b], mode='concat', concat_axis=1)
up_conv_1c = Convolution2D(64, 3, 3, border_mode='same', activation='relu')(merged_1)
up_conv_1d = Convolution2D(64, 3, 3, border_mode='same', activation='relu')(up_conv_1c) 
if debug:
  print("up_conv_1d ", Model(input=input_img, output=up_conv_1d).output_shape )

up_conv_0 = Convolution2D(5, 1, 1, border_mode='same')(up_conv_1d)
if debug:
  print("up_conv_0 ", Model(input=input_img, output=up_conv_0).output_shape )


# reshape to 2D array for softmax
reshaped_conv = Reshape((5, sizeX*sizeY), input_shape=(5, sizeX, sizeY))(up_conv_0)
activated_conv = Activation('softmax')(reshaped_conv)

# ======== Check if model compiles ==========
model = Model(input=input_img, output=activated_conv)
if debug:
  print ("final ", model.output_shape)
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# OneHotLabels = np.array ( [ (np.arange(5) == label2Darray[i][:,:,None]).astype(int) for i in range(label2Darray.shape[0])]  )
OneHotLabels = np.array ( [ ( np.swapaxes(np.swapaxes(((np.arange(5) == label2Darray[i][:,:,None]).astype(int)) , 1,2), 0,1) ).reshape(5, sizeX*sizeY) for i in range(label2Darray.shape[0])]  )
if 0:
  # print (OneHotLabels)
  for i in range(OneHotLabels.shape[0]):
    print (OneHotLabels.sum())


# OneHotLabels[i] = (np.arange(5) == label2Darray[i,:,:]-1).astype(int)
# OneHotLabels =  np.swapaxes(np.swapaxes(OneHotLabels, 1,2), 0,1)

if debug:
  print ('Total slices: ', label2Darray.shape[0])
  print ('Shape of OneHotLabels is: ', OneHotLabels.shape )

model.summary()

if debug:
  print ('Shape of image2Darray is: ', image2Darray.shape[0], image2Darray.shape[1], image2Darray.shape[2])

image2Darray_ = image2Darray.reshape(image2Darray.shape[0], 1, image2Darray.shape[1], image2Darray.shape[2]) 
# image2Darray_ = np.reshape(image2Darray, (image2Darray.shape[0], 1, image2Darray.shape[1], image2Darray.shape[2]) )

if debug:
  print (image2Darray_.shape)

model.fit(image2Darray_, OneHotLabels)
# print (model.output_shape)

# import theano.tensor as T

# def softmax_con(decblock5):
#   sftmxout = T.nnet.softmax(T.transpose(decblock5.reshape((5,400*304))))
#   output = T.argmax((T.transpose(sftmxout)).reshape((1,5,400,304)),axis=1)
#   return output 

# out = softmax_con(up_conv_0)



'''


model.add(Flatten())



model.add(Dense(1024))

model.add(Dropout(0.5))


model.add(Dense(256))


model.add(Dense(5))
model.add(Activation('softmax'))


'''











