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

print (img_array.shape)
print (label_array.shape)

image2Darray = np.array([ img_array[i][j] for i in range(9) for j in range(img_array[i].shape[0]) ])
label2Darray = np.array([ label_array[i][j] for i in range(9) for j in range(label_array[i].shape[0]) ])

import matplotlib.pyplot as plt 

# for i in range(5):
plt.imshow(image2Darray[0])
plt.show()


print (label2Darray[0])
print (label2Darray[0].sum())
print (label2Darray[50].sum())

for i in range(0,img_array[0].shape[0]):
  print (label2Darray[i].sum())













