# ============ File to read mhd images into numpy arrays using sitk===

import SimpleITK as sitk 
import numpy as np 
import glob
from scipy.misc import toimage # to display images for double check
from resizeimage import resizeimage # for image resizing 
import os
from skimage.transform import resize 
import scipy.ndimage as ndi

def ReadData():
  root_dir = '/Users/A/Documents/Fractal/Cartilage/TrainingData-A/'

  img_arr = np.array([ReadImage(root_dir,i) for i in range(1,10)])
  label_arr = np.array([ReadLabel(root_dir,i) for i in range(1,10)])

  return img_arr, label_arr 

def ReadImage(root_dir, i):
  ext = ".mhd"
  imagefilename = os.path.join(root_dir, 'image-{:03d}'.format(i) + ext)
  labelfilename = os.path.join(root_dir, 'labels-{:03d}'.format(i) + ext)

  itk_img = sitk.ReadImage(imagefilename)
  raw_img = sitk.GetArrayFromImage(itk_img) 

  return raw_img

def ReadLabel(root_dir, i):
  ext = ".mhd"
  labelfilename = os.path.join(root_dir, 'labels-{:03d}'.format(i) + ext)

  itk_label = sitk.ReadImage(labelfilename)
  raw_label = sitk.GetArrayFromImage(itk_label)  

  return raw_label

def ContrastStretching(raw_img):
  min_pix, max_pix = p1, p95 = np.percentile(raw_img, (1, 99.5))
  raw_img = (raw_img - min_pix)/(max_pix - min_pix)
  raw_img[raw_img > 1] = 1.
  raw_img[raw_img < 0] = 0.

  return raw_img

def Scaling(raw_img, scale=[1,1,1]):
  raw_spacing = np.array(list(reversed(raw_img.GetSpacing())))
  raw_spacing = np.asarray(raw_spacing)

  spacing = raw_spacing/scale

  scaled_image = ndi.zoom(raw_img, spacing, order = 1)
  scaled_image = scaled_image/np.max(scaled_image)

  return scaled_image


def Resize(raw_img, imageOrLabel= 'image'):
  if imageOrLabel =='label':
    x_dim = 10
    y_dim = 10
  else:
    x_dim = 5
    y_dim = 5
  # print ("x_dim ", x_dim)
  raw_img = resize(raw_img, (x_dim, y_dim) )

  return raw_img

def PreprocessImg(raw_img, imageOrLabel):
  # Contrast streching
  # img_arr = [ContrastStretching(raw_img, imageOrLabel ) for raw_img in img_arr]

  raw_img = ContrastStretching(raw_img)
  raw_img = Scaling(raw_img)
  # raw_img = Resize(raw_img, imageOrLabel)

  return raw_img


def ResizeData(image2DList, ResizedImageList):

  XList =[]
  YList = []
  RatioList = []

  for i in range(0, len(image2DList)):
    XList.append(image2DList[i].shape[0])
    YList.append(image2DList[i].shape[1])
    RatioList.append(image2DList[i].shape[1]/image2DList[i].shape[0])






