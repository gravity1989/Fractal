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









