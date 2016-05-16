# ============ File to read mhd images into numpy arrays using sitk===

# ===== Reorganizing functions=================

import SimpleITK as sitk 
import numpy as np 
import glob
from scipy.misc import toimage # to display images for double check
from resizeimage import resizeimage # for image resizing 
import os
from skimage.transform import resize 
import scipy.ndimage as ndi
import h5py

def ReadData():
  root_dir = '/Users/A/Documents/Fractal/Cartilage/TrainingData-A/'

  img_arr = np.array([ReadAndPreprocessImage(root_dir,i) for i in range(1,10)])
  label_arr = np.array([ReadLabel(root_dir,i) for i in range(1,10)])

  # max X and Y
  maxX, maxY = ResizeData()

  resizedImages = np.array( [ ZeroPad(img, maxX, maxY) for img in img_arr ] )
  resizedLabels = np.array( [ ZeroPad(img, maxX, maxY) for img in label_arr ] )

  return resizedImages, resizedLabels

def ReadAndPreprocessImage(root_dir, i, scale=[1,1,1]):
  ext = ".mhd"
  imagefilename = os.path.join(root_dir, 'image-{:03d}'.format(i) + ext)

  # read itk and convert to array
  itk_img = sitk.ReadImage(imagefilename)
  raw_img = sitk.GetArrayFromImage(itk_img) 

  # Contrast normalization
  min_pix, max_pix = p1, p95 = np.percentile(raw_img, (1, 99.5))
  raw_img = (raw_img - min_pix)/(max_pix - min_pix)
  raw_img[raw_img > 1] = 1.
  raw_img[raw_img < 0] = 0.

  #Scaling 
  raw_spacing = np.array(list(reversed(itk_img.GetSpacing())))
  raw_spacing = np.asarray(raw_spacing)

  spacing = raw_spacing/scale

  scaled_image = ndi.zoom(raw_img, spacing, order = 1)
  scaled_image = scaled_image/np.max(scaled_image)

  return scaled_image


def ReadLabel(root_dir, i, scale=[1,1,1]):
  ext = ".mhd"
  imagefilename = os.path.join(root_dir, 'labels-{:03d}'.format(i) + ext)
  
  # read itk and convert to array
  itk_img = sitk.ReadImage(imagefilename)
  raw_img = sitk.GetArrayFromImage(itk_img) 

  # no normalization needed. Values between 0 and 4 inclusive

  #Scaling
  raw_spacing = np.array(list(reversed(itk_img.GetSpacing())))
  raw_spacing = np.asarray(raw_spacing)

  spacing = raw_spacing/scale

  scaled_image = ndi.zoom(raw_img, spacing, order=1)

  return scaled_image


def ResizeData():

  root_dir = '/Users/A/Documents/Fractal/Cartilage/TrainingData-A/'
  ext = ".mhd"
  img_list = []
  for i in range(1,10):
    imagefilename = os.path.join(root_dir, 'image-{:03d}'.format(i) + ext)
    itk_img = sitk.ReadImage(imagefilename)
    raw_img = sitk.GetArrayFromImage(itk_img) 
    img_list.append(raw_img)

  XList = np.array([img_list[i].shape[1] for i in range(9)])
  YList = np.array([img_list[i].shape[2] for i in range(9)])
  RatioList = np.array( [ img_list[i].shape[2]/img_list[i].shape[1] for i in range(9) ] )

  finalX = XList.max()
  finalY = YList.max()

  return finalX, finalY

import math # for floor and ceiling functions

def ZeroPad(img, maxX, maxY):
  if img.shape[1] < maxX or img.shape[2] < maxY:
    result_img = np.array( [ ZeroPadSlice(img[i], maxX, maxY) for i in range(img.shape[0]) ] )
  else: 
    result_img = img
  return result_img

def ZeroPadSlice (imgSlice, maxX, maxY):
  center = [ imgSlice.shape[0], imgSlice.shape[1]]
  result_img = np.zeros( [maxX, maxY] , dtype = imgSlice.dtype )

  diffX = maxX - imgSlice.shape[0]
  diffY = maxY - imgSlice.shape[1]

  leftStart = 0
  upStart = 0
  if diffX > 0 :
    leftStart = math.floor(diffX/2) 
  if diffY > 0 :
    upStart = math.floor(diffY/2)

  result_img[leftStart:leftStart+imgSlice.shape[0], upStart: upStart+imgSlice.shape[1]] = imgSlice

  return result_img

def ConvertToHdf5():
  val_pats = [9,5]

  images, labels = ReadData()
  root_dir = '/Users/A/Documents/Fractal/Cartilage/TrainingData-A/'
  
  print (root_dir, " : root_dir")

  for i in range(9):
    h5_path_img = os.path.join(root_dir, 'imagehdf5/pat{}.h5'.format(i))
    h5_path_labels = os.path.join(root_dir, 'labelhdf5/pat{}.h5'.format(i))

    print (h5_path_img)
    print (h5_path_labels)
    
    with h5py.File(h5_path_img ,'w') as hf:
        hf.create_dataset('imgs', data=images[i])

    with h5py.File(h5_path_labels,'w') as hf:
        hf.create_dataset('labels', data=labels[i])

def ReadHdf5(debug = 0):
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

  if debug:
    print ("Size of img_array", img_array.shape)
    print ("Size of label_array", label_array.shape)

  return img_array, label_array


def Convert3DTo2D(img_array, label_array, debug = 0):

  noOfRawImages = img_array.shape[0]

  image2Darray = np.array([ img_array[i][j] for i in range(noOfRawImages) for j in range(img_array[i].shape[0]) ])
  label2Darray = np.array([ label_array[i][j] for i in range(noOfRawImages) for j in range(label_array[i].shape[0]) ])

  if (debug):
    print ("Shape of image2Darray is: ", image2Darray.shape)
    print ("Shape of label2Darray is: ", label2Darray.shape)

  return image2Darray, label2Darray

def RemoveZeroLabelSlices (image2Darray, label2Darray, debug=0, threshold =0):

# ========== Ignore label slices with sum = 0 or sum <= threshold ======
  image2Darray = np.array( [image2Darray[i]  for i in range(image2Darray.shape[0]) if (label2Darray[i].sum()>threshold)  ] )
  label2Darray = np.array( [label2Darray[i]  for i in range(label2Darray.shape[0]) if (label2Darray[i].sum()>threshold)  ] )

  if debug:
    print ("Shapes after removing 0 label slices: ")
    print (image2Darray.shape)
    print (label2Darray.shape)

  return image2Darray, label2Darray


def ZeroPadSlicesDivisibleByN (image2Darray, label2Darray, number =16, debug=0):
  # ====== Zero-pad all slices to make size divisible by 16 on both sides (for U-net) ==============
  
  currX = image2Darray.shape[1]
  currY = image2Darray.shape[2]
  desiredX = max (math.floor(currX/16)*16, math.ceil(currX/16)*16)
  desiredY = max (math.floor(currY/16)*16, math.ceil(currY/16)*16)

  image2Darray = np.array( [ZeroPadSlice(image2Darray[i], desiredX, desiredY) for i in range(0, image2Darray.shape[0])] )
  label2Darray = np.array( [ZeroPadSlice(label2Darray[i], desiredX, desiredY) for i in range(0, label2Darray.shape[0])] )
  if debug:
    print ("Shapes after zero padding: ")
    print (image2Darray.shape)
    print (image2Darray.shape[0])
    print (image2Darray.shape[1])
    print (image2Darray[0].shape)
    print (image2Darray[0].shape[0])


  return image2Darray, label2Darray

