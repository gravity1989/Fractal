# Converts mhd images to hdf5 images and labels
# Needs modification to give root directory to the function ConvertToHdf5 
# in order to generalize.

from readData2 import ReadData, ConvertToHdf5

import numpy as np 
import SimpleITK as sitk 

from scipy.misc import toimage # to display images for double check
# toimage(data).show()





# image_arr, label_arr = ReadData()

root_dir = '/Users/A/Documents/Fractal/Cartilage/TrainingData-A/'

ConvertToHdf5(root_dir, 10)


print ("====Sucess=====")

# print (image_arr[0].shape)
# print (image_arr[1].shape)
# print (image_arr[2].shape)
# print (label_arr[0].shape)
# print (label_arr[1].shape)
# print (label_arr[2].shape)



# print (image_arr[0].max())

