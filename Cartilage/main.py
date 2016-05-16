import numpy as np
import h5py
import os 

debug = 0

# Below part reads hdf5 images and stores them in numpy arrays 

root_dir = '/Users/A/Documents/Fractal/Cartilage/TrainingData-A/'
no_entries = 9

img_array = []
label_array = []

for i in range(no_entries):
  h5_path_img = os.path.join(root_dir, 'imagehdf5/pat{}.h5'.format(i))
  h5_path_labels = os.path.join(root_dir, 'labelhdf5/pat{}.h5'.format(i))

  with h5py.File(h5_path_img ,'r') as hf:
    data = hf.get('imgs')
    np_data = np.array(data)
    img_array.append(np_data)
    if (debug):
      print('Shape of image array: \n', img_array[i].shape)

  with h5py.File(h5_path_labels ,'r') as hf:
    data = hf.get('labels')
    np_data = np.array(data)
    label_array.append(np_data)
    if (debug):
      print('Shape of the label array: \n', label_array[i].shape)

img_array = np.array(img_array)
label_array = np.array(label_array)

# print (img_array.shape)
# print (label_array.shape)


# still we have 3D images and labels
# Convert them to 2D slices
image2Darray = np.array([ img_array[i][j] for i in range(9) for j in range(img_array[i].shape[0]) ])
label2Darray = np.array([ label_array[i][j] for i in range(9) for j in range(label_array[i].shape[0]) ])

if (debug):
  print ("Shape of image2Darray is: ", image2Darray.shape)
  print ("Shape of label2Darray is: ", label2Darray.shape)


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














