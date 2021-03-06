# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
try:
  import cPickle as pickle
except ImportError:
  import pickle
import os
import math
from skimage.util import random_noise
from skimage.restoration import (denoise_wavelet, estimate_sigma)
from sklearn.cluster import MiniBatchKMeans
from scipy.stats import uniform
from scipy.stats import gamma
from scipy.stats import rayleigh
import datetime
import tensorflow as tf
import random
from skimage import img_as_float


import matplotlib.pyplot as plt
from utils.timer import Timer
from utils.blob import im_list_to_blob

from model.config import cfg, get_output_dir
from model.bbox_transform import clip_boxes, bbox_transform_inv
from model.nms_wrapper import nms
import wandb
import Automold as am
import Helpers as hp

import matplotlib
from matplotlib.colors import LightSource
from numpy import zeros, newaxis
from PIL import Image
from PIL import ImageEnhance

import os
import subprocess
def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS
  #PIXEL_MEANS = np.array([[[0.36462913, 0.39009895, 0.41216644]]])
  #im_orig -= PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)

def _get_blobs(im):
  """Convert an image and RoIs within that image into network inputs."""
  blobs = {}
  blobs['data'], im_scale_factors = _get_image_blob(im)

  return blobs, im_scale_factors

def _clip_boxes(boxes, im_shape):
  """Clip boxes to image boundaries."""
  # x1 >= 0
  boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
  # y1 >= 0
  boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
  # x2 < im_shape[1]
  boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
  # y2 < im_shape[0]
  boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
  return boxes

def _rescale_boxes(boxes, inds, scales):
  """Rescale boxes according to image rescaling."""
  for i in range(boxes.shape[0]):
    boxes[i,:] = boxes[i,:] / scales[int(inds[i])]

  return boxes

def im_detect(sess, net, im):
  blobs, im_scales = _get_blobs(im)
  assert len(im_scales) == 1, "Only single-image batch implemented"

  im_blob = blobs['data']
  blobs['im_info'] = np.array([im_blob.shape[1], im_blob.shape[2], im_scales[0]], dtype=np.float32)

  _, scores, bbox_pred, rois = net.test_image(sess, blobs['data'], blobs['im_info'])

  
  boxes = rois[:, 1:5] / im_scales[0]
  scores = np.reshape(scores, [scores.shape[0], -1])
  bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])
  if cfg.TEST.BBOX_REG:
    # Apply bounding-box regression deltas
    box_deltas = bbox_pred
    pred_boxes = bbox_transform_inv(boxes, box_deltas)
    pred_boxes = _clip_boxes(pred_boxes, im.shape)
  else:
    # Simply repeat the boxes, once for each class
    pred_boxes = np.tile(boxes, (1, scores.shape[1]))

  return scores, pred_boxes

def apply_nms(all_boxes, thresh):
  """Apply non-maximum suppression to all predicted boxes output by the
  test_net method.
  """
  num_classes = len(all_boxes)
  num_images = len(all_boxes[0])
  nms_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
  for cls_ind in range(num_classes):
    for im_ind in range(num_images):
      dets = all_boxes[cls_ind][im_ind]
      if dets == []:
        continue

      x1 = dets[:, 0]
      y1 = dets[:, 1]
      x2 = dets[:, 2]
      y2 = dets[:, 3]
      scores = dets[:, 4]
      inds = np.where((x2 > x1) & (y2 > y1))[0]
      dets = dets[inds,:]
      if dets == []:
        continue

      keep = nms(dets, thresh)
      if len(keep) == 0:
        continue
      nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
  return nms_boxes



def test_net(sess, net, imdb, weights_filename, noise, max_per_image=100, thresh=0.):
  np.random.seed(cfg.RNG_SEED)
  """Test a Fast R-CNN network on an image database."""
  num_images = len(imdb.image_index)
  # all detections are collected into:
  #  all_boxes[cls][image] = N x 5 array of detections in
  #  (x1, y1, x2, y2, score)
  all_boxes = [[[] for _ in range(num_images)]
         for _ in range(imdb.num_classes)]

  output_dir = get_output_dir(imdb, weights_filename)
  print(output_dir)
  current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  #test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
  test_log_dir = 'logs/gradient_tape/test/' + noise +'/'
  test_summary_writer = tf.summary.FileWriter(test_log_dir)

  #initialize wandb
  wandb.init(project='tf-faster-rcnn-test-ID-testing', entity='mkashyap')

  # timers
  _t = {'im_detect' : Timer(), 'misc' : Timer()}

  for i in range(num_images):
  #for i in range(1):
    img = cv2.imread(imdb.image_path_at(i))
    
    def add_gaussian_noise(noise_type):
        if ('gaussian_wavelet' in noise_type):
            if ('var0.1' in noise_type):
                im_noise = random_noise(img, mode='gaussian', var=0.1)
                im_bayes = denoise_wavelet(im_noise, method='BayesShrink', mode='soft',
                                    wavelet='bior1.5',
                                    multichannel= True, convert2ycbcr=True)
                data = (255 * im_bayes)
                im = data.astype(np.uint8)
                print('gaussian wavelet var 0.1')
            elif ('var1.0' in noise_type):
                im_noise = random_noise(img, mode='gaussian', var=1.0)
                im_bayes = denoise_wavelet(im_noise, method='BayesShrink', mode='soft',
                                    wavelet='bior1.5',
                                    multichannel= True, convert2ycbcr=True)
                data = (255 * im_bayes)
                im = data.astype(np.uint8)
                print('gaussian wavelet var 1.0')
            elif ('var1.5' in noise_type):
                im_noise = random_noise(img, mode='gaussian', var=1.5)
                im_bayes = denoise_wavelet(im_noise, method='BayesShrink', mode='soft',
                                    wavelet='bior1.5',
                                    multichannel= True, convert2ycbcr=True)
                data = (255 * im_bayes)
                im = data.astype(np.uint8)
                print('gaussian wavelet var 1.5')
        elif('gaussian_gaus_blur' in noise_type):
            size = 3
            if ('var0.1' in noise_type):
                gauss_array = random_noise(img, mode='gaussian', var=0.1)
                im_noise = (255 * gauss_array).astype(np.uint8)
                im = cv2.GaussianBlur(im_noise, (size, size), 0)
                print('gaussian blur var 0.1')
            elif ('var1.0' in noise_type):
                gauss_array = random_noise(img, mode='gaussian', var=1.0)
                im_noise = (255 * gauss_array).astype(np.uint8)
                im = cv2.GaussianBlur(im_noise, (size, size), 0)
                print('gaussian blur var 1.0')
            elif ('var1.5' in noise_type):
                gauss_array = random_noise(img, mode='gaussian', var=1.5)
                im_noise = (255 * gauss_array).astype(np.uint8)
                im = cv2.GaussianBlur(im_noise, (size, size), 0)
                print('gaussian blur var 1.5')
        elif('gaussian_mean' in noise_type):
            size = 3
            if ('var0.1' in noise_type):
                gauss_array = random_noise(img, mode='gaussian', var=0.1)
                im_noise = (255 * gauss_array).astype(np.uint8)
                im = cv2.blur(im_noise, (size, size))
                print('gaussian mean var 0.1')
            elif ('var1.0' in noise_type):
                gauss_array = random_noise(img, mode='gaussian', var=1.0)
                im_noise = (255 * gauss_array).astype(np.uint8)
                im = cv2.blur(im_noise, (size, size))
                print('gaussian mean var 1.0')
            elif ('var1.5' in noise_type):
                gauss_array = random_noise(img, mode='gaussian', var=1.5)
                im_noise = (255 * gauss_array).astype(np.uint8)
                im = cv2.blur(im_noise, (size, size))
                print('gaussian mean var 1.5')
        elif('gaussian_median' in noise_type):
            size = 3
            if ('var0.1' in noise_type):
                #gauss_array = random_noise(img, mode='gaussian', var=0.4)
                gauss_array = random_noise(img, mode='gaussian', var=0.1)
                im_noise = (255 * gauss_array).astype(np.uint8)
                im = cv2.medianBlur(im_noise, size)
                print('gaussian median var 0.1')
            elif ('var1.0' in noise_type):
                gauss_array = random_noise(img, mode='gaussian', var=1.0)
                im_noise = (255 * gauss_array).astype(np.uint8)
                im = cv2.medianBlur(im_noise, size)
                print('gaussian median var 1.0')
            elif ('var1.5' in noise_type):
                gauss_array = random_noise(img, mode='gaussian', var=1.5)
                im_noise = (255 * gauss_array).astype(np.uint8)
                im = cv2.medianBlur(im_noise, size)
                print('gaussian median var 1.5')
        elif('gaussian_bilateral' in noise_type):
            diameter = 9      #the diameter of each pixel in the neighborhood used during filtering
            sigmaColor = 20     #sigma of grey/color space.
            sigmaSpace = 100    #Large value means farther pixels influence each other.
            if ('var0.1' in noise_type):
                gauss_array = random_noise(img, mode='gaussian', var=0.1)
                im_noise = (255 * gauss_array).astype(np.uint8)
                im = cv2.bilateralFilter(im_noise, diameter, sigmaColor, sigmaSpace, borderType=cv2.BORDER_CONSTANT)
                print('gaussian bilateral var 0.1')
            elif ('var1.0' in noise_type):
                gauss_array = random_noise(img, mode='gaussian', var=1.0)
                im_noise = (255 * gauss_array).astype(np.uint8)
                im = cv2.bilateralFilter(im_noise, diameter, sigmaColor, sigmaSpace, borderType=cv2.BORDER_CONSTANT)
                print('gaussian bilateral var 1.0')
            elif ('var1.5' in noise_type):
                gauss_array = random_noise(img, mode='gaussian', var=1.5)
                im_noise = (255 * gauss_array).astype(np.uint8)
                im = cv2.bilateralFilter(im_noise, diameter, sigmaColor, sigmaSpace, borderType=cv2.BORDER_CONSTANT)
                print('gaussian bilateral var 1.5')
        else:
            if ('var0.1' in noise_type):
                gauss_array = random_noise(img, mode='gaussian', var=0.1)
                #im = (255 * gauss_array).astype(np.uint8)
                im = gauss_array
                print('gaussian var 0.1')
            elif ('var1.0' in noise_type):
                gauss_array = random_noise(img, mode='gaussian', var=1.0)
                #im = (255 * gauss_array).astype(np.uint8)
                im = gauss_array
                print('gaussian var 1.0')
            elif ('var1.5' in noise_type):
                gauss_array = random_noise(img, mode='gaussian', var=1.5)
                im = gauss_array
                #im = (255 * gauss_array).astype(np.uint8)
                print('gaussian var 1.5')
        print("Gaussian")
        return im 

    def add_poisson_noise(noise_type):
        if ('poisson' in noise_type):
            image = img_as_float(img)
        #introduce poisson noise.
        #also called shot noise originates from the discrete nature of electronic charge or photons.
            if ('poisson_wavelet' in noise_type):
                pois_array = random_noise(img, mode='poisson')
                im_noise = (255 * pois_array).astype(np.uint8)
                im_bayes = denoise_wavelet(im_noise, method='BayesShrink', mode='soft',
                                        wavelet='bior1.5',
                                        multichannel= True, convert2ycbcr=True)
                data = (255 * im_bayes)
                im = data.astype(np.uint8)
                print('poisson wavelet')
            elif('poisson_gaus_blur' in noise_type):
                size = 3
                pois_array = random_noise(img, mode='poisson')
                im_noise = (255 * pois_array).astype(np.uint8)
                im = cv2.GaussianBlur(im_noise, (size, size), 0)
                print('poisson noise with gaussian blur')
            elif('poisson_mean' in noise_type):
                size = 3
                pois_array = random_noise(img, mode='poisson')
                im_noise = (255 * pois_array).astype(np.uint8)
                im = cv2.blur(im_noise, (size, size))
                print('poisson noise with mean filter')
            elif('poisson_median' in noise_type):
                size = 3
                pois_array = random_noise(img, mode='poisson')
                im_noise = (255 * pois_array).astype(np.uint8)
                im = cv2.medianBlur(im_noise, size)
                print('poisson noise with median filter')
            elif('poisson_bilateral' in noise_type):
                diameter = 9      #the diameter of each pixel in the neighborhood used during filtering
                sigmaColor = 20     #sigma of grey/color space.
                sigmaSpace = 100    #Large value means farther pixels influence each other.
                pois_array = random_noise(img, mode='poisson')
                im_noise = (255 * pois_array).astype(np.uint8)
                im = cv2.bilateralFilter(im_noise, diameter, sigmaColor, sigmaSpace, borderType=cv2.BORDER_CONSTANT)
                print('poisson noise with  bilateral filter')
            else:
                pois_array = random_noise(img, mode='poisson')
                #pois_array = np.random.poisson(image)
                #im = pois_array
                im = (255 * pois_array).astype(np.uint8)
                print('poisson noise')
        return im

    def add_sap_noise(noise_type):
        if ('sap' in noise_type):
            if ('sap_wavelet' in noise_type):
                if ('var0.2' in noise_type):
                    sp_array = random_noise(img, mode='s&p', amount=0.2)
                    im_bayes = denoise_wavelet(sp_array, method='BayesShrink', mode='soft',
                                        wavelet='bior1.5',
                                        multichannel= True, convert2ycbcr=True)
                    data = (255 * im_bayes)
                    im = data.astype(np.uint8)
                    print('wavelet')
                    print('s&p wavelet var 0.2')
                elif ('var0.4' in noise_type):
                    sp_array = random_noise(img, mode='s&p', amount=0.4)
                    im_bayes = denoise_wavelet(sp_array, method='BayesShrink', mode='soft',
                                        wavelet='bior1.5',
                                        multichannel= True, convert2ycbcr=True)
                    data = (255 * im_bayes)
                    im = data.astype(np.uint8)
                    print('wavelet')
                    print('s&p wavelet var 0.4')
                elif ('var0.8' in noise_type):
                    sp_array = random_noise(img, mode='s&p', amount=0.8)
                    im_bayes = denoise_wavelet(sp_array, method='BayesShrink', mode='soft',
                                        wavelet='bior1.5',
                                        multichannel= True, convert2ycbcr=True)
                    data = (255 * im_bayes)
                    im = data.astype(np.uint8)
                    print('wavelet')
                    print('s&p wavelet var 0.8')
            elif('sap_gaus_blur' in noise_type):
                size = 3
                if ('var0.2' in noise_type):
                    sp_array = random_noise(img, mode='s&p', amount=0.2)
                    im_noise = (255 * sp_array).astype(np.uint8)
                    im = cv2.GaussianBlur(im_noise, (size, size), 0)
                    print('s&p gaus filter var 0.2')
                elif ('var0.4' in noise_type):
                    sp_array = random_noise(img, mode='s&p', amount=0.4)
                    im_noise = (255 * sp_array).astype(np.uint8)
                    im = cv2.GaussianBlur(im_noise, (size, size), 0)
                    print('s&p gaus filter var 0.4')
                elif ('var0.8' in noise_type):
                    sp_array = random_noise(img, mode='s&p', amount=0.8)
                    im_noise = (255 * sp_array).astype(np.uint8)
                    im = cv2.GaussianBlur(im_noise, (size, size), 0)
                    print('s&p gaus filter var 0.8')
            elif('sap_mean' in noise_type):
                size = 3
                if ('var0.2' in noise_type):
                    sp_array = random_noise(img, mode='s&p', amount=0.2)
                    im_noise = (255 * sp_array).astype(np.uint8)
                    im = cv2.blur(im_noise, (size, size))
                    print('s&p mean var 0.2')
                elif ('var0.4' in noise_type):
                    sp_array = random_noise(img, mode='s&p', amount=0.4)
                    im_noise = (255 * sp_array).astype(np.uint8)
                    im = cv2.blur(im_noise, (size, size))
                    print('s&p mean var 0.4')
                elif ('var0.8' in noise_type):
                    sp_array = random_noise(img, mode='s&p', amount=0.8)
                    im_noise = (255 * sp_array).astype(np.uint8)
                    im = cv2.blur(im_noise, (size, size))
                    print('s&p mean var 0.8')
            elif('sap_median' in noise_type):
                size = 3
                if ('var0.2' in noise_type):
                    sp_array = random_noise(img, mode='s&p', amount=0.2)
                    im_noise = (255 * sp_array).astype(np.uint8)
                    im = cv2.medianBlur(im_noise, size)
                    print('s&p median var 0.2')
                elif ('var0.4' in noise_type):
                    sp_array = random_noise(img, mode='s&p', amount=0.4)
                    im_noise = (255 * sp_array).astype(np.uint8)
                    im = cv2.medianBlur(im_noise, size)
                    print('s&p median var 0.4')
                elif ('var0.8' in noise_type):
                    sp_array = random_noise(img, mode='s&p', amount=0.8)
                    im_noise = (255 * sp_array).astype(np.uint8)
                    im = cv2.medianBlur(im_noise, size)
                    print('s&p median var 0.8')
            elif('sap_bilateral' in noise_type):
                diameter = 9      #the diameter of each pixel in the neighborhood used during filtering
                sigmaColor = 20     #sigma of grey/color space.
                sigmaSpace = 100    #Large value means farther pixels influence each other.
                if ('var0.2' in noise_type):
                    sp_array = random_noise(img, mode='s&p', amount=0.2)
                    im_noise = (255 * sp_array).astype(np.uint8)
                    im = cv2.bilateralFilter(im_noise, diameter, sigmaColor, sigmaSpace, borderType=cv2.BORDER_CONSTANT)
                    print('s&p bilateral var 0.2')
                elif ('var0.4' in noise_type):
                    sp_array = random_noise(img, mode='s&p', amount=0.4)
                    im_noise = (255 * sp_array).astype(np.uint8)
                    im = cv2.bilateralFilter(im_noise, diameter, sigmaColor, sigmaSpace, borderType=cv2.BORDER_CONSTANT)
                    print('s&p bilateral var 0.4')
                elif ('var0.8' in noise_type):
                    sp_array = random_noise(img, mode='s&p', amount=0.8)
                    im_noise = (255 * sp_array).astype(np.uint8)
                    im = cv2.bilateralFilter(im_noise, diameter, sigmaColor, sigmaSpace, borderType=cv2.BORDER_CONSTANT)
                    print('s&p bilateral var 0.8')
            else:
                if ('var0.2' in noise_type):
                    sp_array = random_noise(img, mode='s&p', amount=0.2)
                    #im = sp_array
                    im = (255 * sp_array).astype(np.uint8)
                    print('s&p var 0.2')
                elif ('var0.4' in noise_type):
                    sp_array = random_noise(img, mode='s&p', amount=0.4)
                    #im = sp_array
                    im = (255 * sp_array).astype(np.uint8)
                    print('s&p var 0.4')
                elif ('var0.8' in noise_type):
                    sp_array = random_noise(img, mode='s&p', amount=0.8)
                    #im = sp_array
                    im = (255 * sp_array).astype(np.uint8)
                    print('s&p var 0.8')
            print("salt & pepper")
        return im

    def add_speckle_noise(noise_type):
        if ('speckle' in noise_type):
            if ('speckle_wavelet' in noise_type):
                if ('var0.5' in noise_type):
                    speck_array = random_noise(img, mode='speckle', var=0.5)
                    im_bayes = denoise_wavelet(speck_array, method='BayesShrink', mode='soft',
                                        wavelet='bior1.5',
                                        multichannel= True, convert2ycbcr=True)
                    data = (255 * im_bayes)
                    im = data.astype(np.uint8)
                    print('speckle wavelet var 0.4')
                elif ('var1.0' in noise_type):
                    speck_array = random_noise(img, mode='speckle', var=1.0)
                    im_bayes = denoise_wavelet(speck_array, method='BayesShrink', mode='soft',
                                        wavelet='bior1.5',
                                        multichannel= True, convert2ycbcr=True)
                    data = (255 * im_bayes)
                    im = data.astype(np.uint8)
                    print('speckle wavelet var 1.0')
                elif ('var2.0' in noise_type):
                    speck_array = random_noise(img, mode='speckle', var=2.0)
                    im_bayes = denoise_wavelet(speck_array, method='BayesShrink', mode='soft',
                                        wavelet='bior1.5',
                                        multichannel= True, convert2ycbcr=True)
                    data = (255 * im_bayes)
                    im = data.astype(np.uint8)
                    print('speckle wavelet var 2.0')
            elif('speckle_gaus_blur' in noise_type):
                size = 3
                if ('var0.5' in noise_type):
                    speck_array = random_noise(img, mode='speckle', var=0.5)
                    im_noise = (255 * speck_array).astype(np.uint8)
                    im = cv2.GaussianBlur(im_noise, (size, size), 0)
                    print('speckle gaus blur var 0.5')
                elif ('var1.0' in noise_type):
                    speck_array = random_noise(img, mode='speckle', var=1.0)
                    im_noise = (255 * speck_array).astype(np.uint8)
                    im = cv2.GaussianBlur(im_noise, (size, size), 0)
                    print('speckle gaus blur 1.0')
                elif ('var2.0' in noise_type):
                    speck_array = random_noise(img, mode='speckle', var=2.0)
                    im_noise = (255 * speck_array).astype(np.uint8)
                    im = cv2.GaussianBlur(im_noise, (size, size), 0)
                    print('speckle gaus blur 2.0')
            elif('speckle_mean' in noise_type):
                size = 3
                if ('var0.5' in noise_type):
                    speck_array = random_noise(img, mode='speckle', var=0.5)
                    im_noise = (255 * speck_array).astype(np.uint8)
                    im = cv2.blur(im_noise, (size, size))
                    print('speckle mean var 0.5')
                elif ('var1.0' in noise_type):
                    speck_array = random_noise(img, mode='speckle', var=1.0)
                    im_noise = (255 * speck_array).astype(np.uint8)
                    im = cv2.blur(im_noise, (size, size))
                    print('speckle mean 1.0')
                elif ('var2.0' in noise_type):
                    speck_array = random_noise(img, mode='speckle', var=2.0)
                    im_noise = (255 * speck_array).astype(np.uint8)
                    im = cv2.blur(im_noise, (size, size))
                    print('speckle mean 2.0')
            elif('speckle_median' in noise_type):
                size = 3
                if ('var0.5' in noise_type):
                    speck_array = random_noise(img, mode='speckle', var=0.5)
                    im_noise = (255 * speck_array).astype(np.uint8)
                    im = cv2.medianBlur(im_noise, size)
                    print('speckle median var 0.5')
                elif ('var1.0' in noise_type):
                    speck_array = random_noise(img, mode='speckle', var=1.0)
                    im_noise = (255 * speck_array).astype(np.uint8)
                    im = cv2.medianBlur(im_noise, size)
                    print('speckle median 1.0')
                elif ('var2.0' in noise_type):
                    speck_array = random_noise(img, mode='speckle', var=2.0)
                    im_noise = (255 * speck_array).astype(np.uint8)
                    im = cv2.medianBlur(im_noise, size)
                    print('speckle median 2.0')
            elif('speckle_bilateral' in noise_type):
                diameter = 9      #the diameter of each pixel in the neighborhood used during filtering
                sigmaColor = 20     #sigma of grey/color space.
                sigmaSpace = 100    #Large value means farther pixels influence each other.
                if ('var0.5' in noise_type):
                    speck_array = random_noise(img, mode='speckle', var=0.5)
                    im_noise = (255 * speck_array).astype(np.uint8)
                    im = cv2.bilateralFilter(im_noise, diameter, sigmaColor, sigmaSpace, borderType=cv2.BORDER_CONSTANT)
                    print('speckle bilateral var 0.5')
                elif ('var1.0' in noise_type):
                    speck_array = random_noise(img, mode='speckle', var=1.0)
                    im_noise = (255 * speck_array).astype(np.uint8)
                    im = cv2.bilateralFilter(im_noise, diameter, sigmaColor, sigmaSpace, borderType=cv2.BORDER_CONSTANT)
                    print('speckle bilateral 1.0')
                elif ('var2.0' in noise_type):
                    speck_array = random_noise(img, mode='speckle', var=2.0)
                    im_noise = (255 * speck_array).astype(np.uint8)
                    im = cv2.bilateralFilter(im_noise, diameter, sigmaColor, sigmaSpace, borderType=cv2.BORDER_CONSTANT)
                    print('speckle bilateral 2.0')
            else:
                if ('var0.5' in noise_type):
                    speck_array = random_noise(img, mode='speckle', var=0.5)
                    #im = speck_array
                    im = (255 * speck_array).astype(np.uint8)
                    print('speckle var 0.5')
                elif ('var1.0' in noise_type):
                    speck_array = random_noise(img, mode='speckle', var=1.0)
                    #im = speck_array
                    im = (255 * speck_array).astype(np.uint8)
                    print('speckle var 1.0')
                elif ('var2.0' in noise_type):
                    speck_array = random_noise(img, mode='speckle', var=2.0)
                    #im = speck_array 
                    im = (255 * speck_array).astype(np.uint8)
                    print('speckle var 2.0')
            print("Speckle")
        return im

    def add_quant_noise(noise_type):
        if ('quant' in noise_type):
            img1 = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            h, w = img1.shape[:2]
            #clor quantization, using K-Means clustering.
            #Usually this noise is found while converting analog to digital, or continuous random variable to discreate.
            image = img1.reshape((img1.shape[0] * img1.shape[1], 3))
            if ('quant_wavelet' in noise_type):
                if ('var3' in noise_type):
                    clt = MiniBatchKMeans(n_clusters= 3)
                    labels = clt.fit_predict(image)
                    quant = clt.cluster_centers_.astype("uint8")[labels]
                    quant = quant.reshape(h, w, 3)
                    quant_array = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
                    im_bayes = denoise_wavelet(quant_array, method='BayesShrink', mode='soft',
                                        wavelet='bior1.5',
                                        multichannel= True, convert2ycbcr=True)
                    data = (255 * im_bayes)
                    im = data.astype(np.uint8)
                    print('quantization wavelet with cluster 3')
                elif ('var7' in noise_type):
                    clt = MiniBatchKMeans(n_clusters= 7)
                    labels = clt.fit_predict(image)
                    quant = clt.cluster_centers_.astype("uint8")[labels]
                    quant = quant.reshape(h, w, 3)
                    quant_array = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
                    im_bayes = denoise_wavelet(quant_array, method='BayesShrink', mode='soft',
                                        wavelet='bior1.5',
                                        multichannel= True, convert2ycbcr=True)
                    data = (255 * im_bayes)
                    im = data.astype(np.uint8)
                    print('quantization wavelet with cluster 7')
                elif ('var10' in noise_type):
                    clt = MiniBatchKMeans(n_clusters= 10)
                    labels = clt.fit_predict(image)
                    quant = clt.cluster_centers_.astype("uint8")[labels]
                    quant = quant.reshape(h, w, 3)
                    quant_array = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
                    im_bayes = denoise_wavelet(quant_array, method='BayesShrink', mode='soft',
                                        wavelet='bior1.5',
                                        multichannel= True, convert2ycbcr=True)
                    data = (255 * im_bayes)
                    im = data.astype(np.uint8)
                    print('quantization wavelet with cluster 10')
            elif('quant_gaus_blur' in noise_type):
                size = 3
                if ('var3' in noise_type):
                    clt = MiniBatchKMeans(n_clusters= 3)
                    labels = clt.fit_predict(image)
                    quant = clt.cluster_centers_.astype("uint8")[labels]
                    quant = quant.reshape(h, w, 3)
                    im_noise = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
                    im = cv2.GaussianBlur(im_noise, (size, size), 0)
                    print('quantization gausblur with cluster 3')
                elif ('var7' in noise_type):
                    clt = MiniBatchKMeans(n_clusters= 7)
                    labels = clt.fit_predict(image)
                    quant = clt.cluster_centers_.astype("uint8")[labels]
                    quant = quant.reshape(h, w, 3)
                    im_noise = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
                    im = cv2.GaussianBlur(im_noise, (size, size), 0)
                    print('quantization gasublur with cluster 7')
                elif ('var10' in noise_type):
                    clt = MiniBatchKMeans(n_clusters= 10)
                    labels = clt.fit_predict(image)
                    quant = clt.cluster_centers_.astype("uint8")[labels]
                    quant = quant.reshape(h, w, 3)
                    im_noise = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
                    im = cv2.GaussianBlur(im_noise, (size, size), 0)
                    print('quantization gausblur with cluster 10')
            elif('quant_mean' in noise_type):
                size = 3
                if ('var3' in noise_type):
                    clt = MiniBatchKMeans(n_clusters= 3)
                    labels = clt.fit_predict(image)
                    quant = clt.cluster_centers_.astype("uint8")[labels]
                    quant = quant.reshape(h, w, 3)
                    im_noise = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
                    im = cv2.blur(im_noise, (size, size))
                    print('quantization mean with cluster 3')
                elif ('var7' in noise_type):
                    clt = MiniBatchKMeans(n_clusters= 7)
                    labels = clt.fit_predict(image)
                    quant = clt.cluster_centers_.astype("uint8")[labels]
                    quant = quant.reshape(h, w, 3)
                    im_noise = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
                    im = cv2.blur(im_noise, (size, size))
                    print('quantization mean with cluster 7')
                elif ('var10' in noise_type):
                    clt = MiniBatchKMeans(n_clusters= 10)
                    labels = clt.fit_predict(image)
                    quant = clt.cluster_centers_.astype("uint8")[labels]
                    quant = quant.reshape(h, w, 3)
                    im_noise = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
                    im = cv2.blur(im_noise, (size, size))
                    print('quantization mean with cluster 10')
            elif('quant_median' in noise_type):
                size = 3
                if ('var3' in noise_type):
                    clt = MiniBatchKMeans(n_clusters= 3)
                    labels = clt.fit_predict(image)
                    quant = clt.cluster_centers_.astype("uint8")[labels]
                    quant = quant.reshape(h, w, 3)
                    im_noise = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
                    im = cv2.medianBlur(im_noise, size)
                    print('quantization median with cluster 3')
                elif ('var7' in noise_type):
                    clt = MiniBatchKMeans(n_clusters= 7)
                    labels = clt.fit_predict(image)
                    quant = clt.cluster_centers_.astype("uint8")[labels]
                    quant = quant.reshape(h, w, 3)
                    im_noise = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
                    im = cv2.medianBlur(im_noise, size)
                    print('quantization median with cluster 7')
                elif ('var10' in noise_type):
                    clt = MiniBatchKMeans(n_clusters= 10)
                    labels = clt.fit_predict(image)
                    quant = clt.cluster_centers_.astype("uint8")[labels]
                    quant = quant.reshape(h, w, 3)
                    im_noise = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
                    im = cv2.medianBlur(im_noise, size)
                    print('quantization median with cluster 10')
            elif('quant_bilateral' in noise_type):
                diameter = 9      #the diameter of each pixel in the neighborhood used during filtering
                sigmaColor = 20     #sigma of grey/color space.
                sigmaSpace = 100    #Large value means farther pixels influence each other.
                if ('var3' in noise_type):
                    clt = MiniBatchKMeans(n_clusters= 3)
                    labels = clt.fit_predict(image)
                    quant = clt.cluster_centers_.astype("uint8")[labels]
                    quant = quant.reshape(h, w, 3)
                    im_noise = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
                    im = cv2.bilateralFilter(im_noise, diameter, sigmaColor, sigmaSpace, borderType=cv2.BORDER_CONSTANT)
                    print('quantization bilateral with cluster 3')
                elif ('var7' in noise_type):
                    clt = MiniBatchKMeans(n_clusters= 7)
                    labels = clt.fit_predict(image)
                    quant = clt.cluster_centers_.astype("uint8")[labels]
                    quant = quant.reshape(h, w, 3)
                    im_noise = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
                    im = cv2.bilateralFilter(im_noise, diameter, sigmaColor, sigmaSpace, borderType=cv2.BORDER_CONSTANT)
                    print('quantization bilateral with cluster 7')
                elif ('var10' in noise_type):
                    clt = MiniBatchKMeans(n_clusters= 10)
                    labels = clt.fit_predict(image)
                    quant = clt.cluster_centers_.astype("uint8")[labels]
                    quant = quant.reshape(h, w, 3)
                    im_noise = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
                    im = cv2.bilateralFilter(im_noise, diameter, sigmaColor, sigmaSpace, borderType=cv2.BORDER_CONSTANT)
                    print('quantization bilateral with cluster 10')
            else:
                if ('var3' in noise_type):
                    clt = MiniBatchKMeans(n_clusters= 3)
                    labels = clt.fit_predict(image)
                    quant = clt.cluster_centers_.astype("uint8")[labels]
                    quant = quant.reshape(h, w, 3)
                    im = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
                    print('quantization with cluster 3')
                elif ('var7' in noise_type):
                    clt = MiniBatchKMeans(n_clusters= 7)
                    labels = clt.fit_predict(image)
                    quant = clt.cluster_centers_.astype("uint8")[labels]
                    quant = quant.reshape(h, w, 3)
                    im = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
                    print('quantization with cluster 7')
                elif ('var10' in noise_type):
                    clt = MiniBatchKMeans(n_clusters= 10)
                    labels = clt.fit_predict(image)
                    quant = clt.cluster_centers_.astype("uint8")[labels]
                    quant = quant.reshape(h, w, 3)
                    im = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
                    print('quantization with cluster 10')
            print("Quantization")
        return im

    def add_uniform_noise(noise_type):
        if('uniform' in noise_type):
            image = img_as_float(img)
            if ('uniform_wavelet' in noise_type):
                if ('var0.2' in noise_type):
                    uniform_array = np.random.uniform(low=0., high=0.2, size=img.shape)
                    im_noise = cv2.add(image, uniform_array)
                    im_bayes = denoise_wavelet(im_noise, method='BayesShrink', mode='soft',
                                        wavelet='bior1.5',
                                        multichannel= True, convert2ycbcr=True)
                    data = (255 * im_bayes)
                    im = data.astype(np.uint8)
                    print('wavelet')
                    print('uniform wavelet var 0.2')
                elif ('var0.6' in noise_type):
                    uniform_array = np.random.uniform(low=0., high=0.6, size=img.shape)
                    im_noise = cv2.add(image, uniform_array)
                    im_bayes = denoise_wavelet(im_noise, method='BayesShrink', mode='soft',
                                        wavelet='bior1.5',
                                        multichannel= True, convert2ycbcr=True)
                    data = (255 * im_bayes)
                    im = data.astype(np.uint8)
                    print('wavelet')
                    print('uniform wavelet var 0.6')
                elif ('var1.2' in noise_type):
                    uniform_array = np.random.uniform(low=0., high=1.2, size=img.shape)
                    im_noise = cv2.add(image, uniform_array)
                    im_bayes = denoise_wavelet(im_noise, method='BayesShrink', mode='soft',
                                        wavelet='bior1.5',
                                        multichannel= True, convert2ycbcr=True)
                    data = (255 * im_bayes)
                    im = data.astype(np.uint8)
                    print('wavelet')
                    print('uniform wavelet var 1.2')
            elif('uniform_gaus_blur' in noise_type):
                size = 3
                if ('var0.2' in noise_type):
                    uniform_array = np.random.uniform(low=0., high=0.2, size=img.shape)
                    uniform_noise = cv2.add(image, uniform_array)
                    im_noise = (255 * uniform_noise).astype(np.uint8)
                    im = cv2.GaussianBlur(im_noise, (size, size), 0)
                    print('uniform var 0.2')
                elif ('var0.6' in noise_type):
                    uniform_array = np.random.uniform(low=0., high=0.6, size=img.shape)
                    uniform_noise = cv2.add(image, uniform_array)
                    im_noise = (255 * uniform_noise).astype(np.uint8)
                    im = cv2.GaussianBlur(im_noise, (size, size), 0)
                    print('uniform var 0.6')
                elif ('var1.2' in noise_type):
                    uniform_array = np.random.uniform(low=0., high=1.2, size=img.shape)
                    uniform_noise = cv2.add(image, uniform_array)
                    im_noise = (255 * uniform_noise).astype(np.uint8)
                    im = cv2.GaussianBlur(im_noise, (size, size), 0)
                    print('uniform var 1.2')
            elif('uniform_mean' in noise_type):
                size = 3
                if ('var0.2' in noise_type):
                    uniform_array = np.random.uniform(low=0., high=0.2, size=img.shape)
                    uniform_noise = cv2.add(image, uniform_array)
                    im_noise = (255 * uniform_noise).astype(np.uint8)
                    im = cv2.blur(im_noise, (size, size))
                    print('uniform mean var 0.2')
                elif ('var0.6' in noise_type):
                    uniform_array = np.random.uniform(low=0., high=0.6, size=img.shape)
                    uniform_noise = cv2.add(image, uniform_array)
                    im_noise = (255 * uniform_noise).astype(np.uint8)
                    im = cv2.blur(im_noise, (size, size))
                    print('uniform mean var 0.6')
                elif ('var1.2' in noise_type):
                    uniform_array = np.random.uniform(low=0., high=1.2, size=img.shape)
                    uniform_noise = cv2.add(image, uniform_array)
                    im_noise = (255 * uniform_noise).astype(np.uint8)
                    im = cv2.blur(im_noise, (size, size))
                    print('uniform mean var 1.2')
            elif('uniform_median' in noise_type):
                size = 3
                if ('var0.2' in noise_type):
                    uniform_array = np.random.uniform(low=0., high=0.2, size=img.shape)
                    uniform_noise = cv2.add(image, uniform_array)
                    im_noise = (255 * uniform_noise).astype(np.uint8)
                    im = cv2.medianBlur(im_noise, size)
                    print('uniform median var 0.2')
                elif ('var0.6' in noise_type):
                    uniform_array = np.random.uniform(low=0., high=0.6, size=img.shape)
                    uniform_noise = cv2.add(image, uniform_array)
                    im_noise = (255 * uniform_noise).astype(np.uint8)
                    im = cv2.medianBlur(im_noise, size)
                    print('uniform median var 0.6')
                elif ('var1.2' in noise_type):
                    uniform_array = np.random.uniform(low=0., high=1.2, size=img.shape)
                    uniform_noise = cv2.add(image, uniform_array)
                    im_noise = (255 * uniform_noise).astype(np.uint8)
                    im = cv2.medianBlur(im_noise, size)
                    print('uniform median var 1.2')
            elif('uniform_bilateral' in noise_type):
                diameter = 9      #the diameter of each pixel in the neighborhood used during filtering
                sigmaColor = 20     #sigma of grey/color space.
                sigmaSpace = 100    #Large value means farther pixels influence each other.
                if ('var0.2' in noise_type):
                    uniform_array = np.random.uniform(low=0., high=0.2, size=img.shape)
                    uniform_noise = cv2.add(image, uniform_array)
                    im_noise = (255 * uniform_noise).astype(np.uint8)
                    im = cv2.bilateralFilter(im_noise, diameter, sigmaColor, sigmaSpace, borderType=cv2.BORDER_CONSTANT)
                    print('uniform bilateral var 0.2')
                elif ('var0.6' in noise_type):
                    uniform_array = np.random.uniform(low=0., high=0.6, size=img.shape)
                    uniform_noise = cv2.add(image, uniform_array)
                    im_noise = (255 * uniform_noise).astype(np.uint8)
                    im = cv2.bilateralFilter(im_noise, diameter, sigmaColor, sigmaSpace, borderType=cv2.BORDER_CONSTANT)
                    print('uniform bilateral var 0.6')
                elif ('var1.2' in noise_type):
                    uniform_array = np.random.uniform(low=0., high=1.2, size=img.shape)
                    uniform_noise = cv2.add(image, uniform_array)
                    im_noise = (255 * uniform_noise).astype(np.uint8)
                    im = cv2.bilateralFilter(im_noise, diameter, sigmaColor, sigmaSpace, borderType=cv2.BORDER_CONSTANT)
                    print('uniform bilateral var 1.2')
            else:
                if ('var0.2' in noise_type):
                    uniform_array = np.random.uniform(low=0., high=0.2, size=img.shape)
                    uniform_noise = cv2.add(image, uniform_array)
                    #im = uniform_noise
                    im = (255 * uniform_noise).astype(np.uint8)
                    print('uniform var 0.2')
                elif ('var0.6' in noise_type):
                    uniform_array = np.random.uniform(low=0., high=0.6, size=img.shape)
                    uniform_noise = cv2.add(image, uniform_array)
                    #im = uniform_noise
                    im = (255 * uniform_noise).astype(np.uint8)
                    print('uniform var 0.6')
                elif ('var1.2' in noise_type):
                    uniform_array = np.random.uniform(low=0., high=1.2, size=img.shape)
                    uniform_noise = cv2.add(image, uniform_array)
                    #im = uniform_noise
                    im = (255 * uniform_noise).astype(np.uint8)
                    print('uniform var 1.2')
            print("uniform")
        return im

    def add_brownian_noise(noise_type):
        if ('brownian' in noise_type):
            h, w = img.shape[:2]
            n=img.size
            #T=n
            #times = np.linspace(0., T, n)
            #dt = times[1] - times[0]
            #Bt2 - Bt1 ~ Normal with mean 0 and variance t2-t1
            #brownian motion's characterstics is its independent normally distributed increments.
            if ('brownian_wavelet' in noise_type):
                if ('var0.9' in noise_type):
                    dt = 0.9
                    dB = np.sqrt(dt) * np.random.normal(size=(n-1,))
                    B0 = np.zeros(shape=(1,))
                    B = np.concatenate((B0, np.cumsum(dB)))
                    brownian = (B * 255).astype(np.uint8)
                    brownian_noise = brownian.reshape(h,w,3)
                    im_noise = cv2.add(img, brownian_noise)
                    im_bayes = denoise_wavelet(im_noise, method='BayesShrink', mode='soft',
                                        wavelet='bior1.5',
                                        multichannel= True, convert2ycbcr=True)
                    data = (255 * im_bayes)
                    im = data.astype(np.uint8)
                    print('brownian wavelet var 0.9')
                elif ('var0.09' in noise_type):
                    dt = 0.09
                    dB = np.sqrt(dt) * np.random.normal(size=(n-1,))
                    B0 = np.zeros(shape=(1,))
                    B = np.concatenate((B0, np.cumsum(dB)))
                    brownian = (B * 255).astype(np.uint8)
                    brownian_noise = brownian.reshape(h,w,3)
                    im_noise = cv2.add(img, brownian_noise)
                    im_bayes = denoise_wavelet(im_noise, method='BayesShrink', mode='soft',
                                        wavelet='bior1.5',
                                        multichannel= True, convert2ycbcr=True)
                    data = (255 * im_bayes)
                    im = data.astype(np.uint8)
                    print('brownian wavelet var 0.09')
                elif ('var0.009' in noise_type):
                    dt = 0.009
                    dB = np.sqrt(dt) * np.random.normal(size=(n-1,))
                    B0 = np.zeros(shape=(1,))
                    B = np.concatenate((B0, np.cumsum(dB)))
                    brownian = (B * 255).astype(np.uint8)
                    brownian_noise = brownian.reshape(h,w,3)
                    im_noise = cv2.add(img, brownian_noise)
                    im_bayes = denoise_wavelet(im_noise, method='BayesShrink', mode='soft',
                                        wavelet='bior1.5',
                                        multichannel= True, convert2ycbcr=True)
                    data = (255 * im_bayes)
                    im = data.astype(np.uint8)
                    print('brownian wavelet var 0.009')
            elif('brownian_gaus_blur' in noise_type):
                size = 3
                if ('var0.9' in noise_type):
                    dt = 0.9
                    dB = np.sqrt(dt) * np.random.normal(size=(n-1,))
                    #brownian motion starts at zero
                    B0 = np.zeros(shape=(1,))
                    #brownian motion is to concatenate the intial value with the cumulative sum of the increments.
                    B = np.concatenate((B0, np.cumsum(dB)))
                    brownian = (B * 255).astype(np.uint8)
                    brownian_noise = brownian.reshape(h,w,3)
                    im_noise = cv2.add(img, brownian_noise)
                    im = cv2.GaussianBlur(im_noise, (size, size), 0)
                    print('brownian gausblur 0.9')
                elif ('var0.09' in noise_type):
                    dt = 0.09
                    dB = np.sqrt(dt) * np.random.normal(size=(n-1,))
                    B0 = np.zeros(shape=(1,))
                    B = np.concatenate((B0, np.cumsum(dB)))
                    brownian = (B * 255).astype(np.uint8)
                    brownian_noise = brownian.reshape(h,w,3)
                    im_noise = cv2.add(img, brownian_noise)
                    im = cv2.GaussianBlur(im_noise, (size, size), 0)
                    print('brownian gausblur 0.09')
                elif ('var0.009' in noise_type):
                    dt = 0.009
                    dB = np.sqrt(dt) * np.random.normal(size=(n-1,))
                    B0 = np.zeros(shape=(1,))
                    B = np.concatenate((B0, np.cumsum(dB)))
                    brownian = (B * 255).astype(np.uint8)
                    brownian_noise = brownian.reshape(h,w,3)
                    im_noise = cv2.add(img, brownian_noise)
                    im = cv2.GaussianBlur(im_noise, (size, size), 0)
                    print('brownian gausblur 0.009')
            elif('brownian_mean' in noise_type):
                size = 3
                if ('var0.9' in noise_type):
                    dt = 0.9
                    dB = np.sqrt(dt) * np.random.normal(size=(n-1,))
                    #brownian motion starts at zero
                    B0 = np.zeros(shape=(1,))
                    #brownian motion is to concatenate the intial value with the cumulative sum of the increments.
                    B = np.concatenate((B0, np.cumsum(dB)))
                    brownian = (B * 255).astype(np.uint8)
                    brownian_noise = brownian.reshape(h,w,3)
                    im_noise = cv2.add(img, brownian_noise)
                    im = cv2.blur(im_noise, (size, size))
                    print('brownian mean 0.9')
                elif ('var0.09' in noise_type):
                    dt = 0.09
                    dB = np.sqrt(dt) * np.random.normal(size=(n-1,))
                    B0 = np.zeros(shape=(1,))
                    B = np.concatenate((B0, np.cumsum(dB)))
                    brownian = (B * 255).astype(np.uint8)
                    brownian_noise = brownian.reshape(h,w,3)
                    im_noise = cv2.add(img, brownian_noise)
                    im = cv2.blur(im_noise, (size, size))
                    print('brownian mean 0.09')
                elif ('var0.009' in noise_type):
                    dt = 0.009
                    dB = np.sqrt(dt) * np.random.normal(size=(n-1,))
                    B0 = np.zeros(shape=(1,))
                    B = np.concatenate((B0, np.cumsum(dB)))
                    brownian = (B * 255).astype(np.uint8)
                    brownian_noise = brownian.reshape(h,w,3)
                    im_noise = cv2.add(img, brownian_noise)
                    im = cv2.blur(im_noise, (size, size))
                    print('brownian mean 0.009')
            elif('brownian_median' in noise_type):
                size = 3
                if ('var0.9' in noise_type):
                    dt = 0.9
                    dB = np.sqrt(dt) * np.random.normal(size=(n-1,))
                    #brownian motion starts at zero
                    B0 = np.zeros(shape=(1,))
                    #brownian motion is to concatenate the intial value with the cumulative sum of the increments.
                    B = np.concatenate((B0, np.cumsum(dB)))
                    brownian = (B * 255).astype(np.uint8)
                    brownian_noise = brownian.reshape(h,w,3)
                    im_noise = cv2.add(img, brownian_noise)
                    im = cv2.medianBlur(im_noise, size)
                    print('brownian median 0.9')
                elif ('var0.09' in noise_type):
                    dt = 0.09
                    dB = np.sqrt(dt) * np.random.normal(size=(n-1,))
                    B0 = np.zeros(shape=(1,))
                    B = np.concatenate((B0, np.cumsum(dB)))
                    brownian = (B * 255).astype(np.uint8)
                    brownian_noise = brownian.reshape(h,w,3)
                    im_noise = cv2.add(img, brownian_noise)
                    im = cv2.medianBlur(im_noise, size)
                    print('brownian median 0.09')
                elif ('var0.009' in noise_type):
                    dt = 0.009
                    dB = np.sqrt(dt) * np.random.normal(size=(n-1,))
                    B0 = np.zeros(shape=(1,))
                    B = np.concatenate((B0, np.cumsum(dB)))
                    brownian = (B * 255).astype(np.uint8)
                    brownian_noise = brownian.reshape(h,w,3)
                    im_noise = cv2.add(img, brownian_noise)
                    im = cv2.medianBlur(im_noise, size)
                    print('brownian median 0.009')
            elif('brownian_bilateral' in noise_type):
                diameter = 9      #the diameter of each pixel in the neighborhood used during filtering
                sigmaColor = 20     #sigma of grey/color space.
                sigmaSpace = 100    #Large value means farther pixels influence each other.
                if ('var0.9' in noise_type):
                    dt = 0.9
                    dB = np.sqrt(dt) * np.random.normal(size=(n-1,))
                    #brownian motion starts at zero
                    B0 = np.zeros(shape=(1,))
                    #brownian motion is to concatenate the intial value with the cumulative sum of the increments.
                    B = np.concatenate((B0, np.cumsum(dB)))
                    brownian = (B * 255).astype(np.uint8)
                    brownian_noise = brownian.reshape(h,w,3)
                    im_noise = cv2.add(img, brownian_noise)
                    im = cv2.bilateralFilter(im_noise, diameter, sigmaColor, sigmaSpace, borderType=cv2.BORDER_CONSTANT)
                    print('brownian bilateral 0.9')
                elif ('var0.09' in noise_type):
                    dt = 0.09
                    dB = np.sqrt(dt) * np.random.normal(size=(n-1,))
                    B0 = np.zeros(shape=(1,))
                    B = np.concatenate((B0, np.cumsum(dB)))
                    brownian = (B * 255).astype(np.uint8)
                    brownian_noise = brownian.reshape(h,w,3)
                    im_noise = cv2.add(img, brownian_noise)
                    im = cv2.bilateralFilter(im_noise, diameter, sigmaColor, sigmaSpace, borderType=cv2.BORDER_CONSTANT)
                    print('brownian bilateral 0.09')
                elif ('var0.009' in noise_type):
                    dt = 0.009
                    dB = np.sqrt(dt) * np.random.normal(size=(n-1,))
                    B0 = np.zeros(shape=(1,))
                    B = np.concatenate((B0, np.cumsum(dB)))
                    brownian = (B * 255).astype(np.uint8)
                    brownian_noise = brownian.reshape(h,w,3)
                    im_noise = cv2.add(img, brownian_noise)
                    im = cv2.bilateralFilter(im_noise, diameter, sigmaColor, sigmaSpace, borderType=cv2.BORDER_CONSTANT)
                    print('brownian bilateral 0.009')
            else:
                if ('var0.9' in noise_type):
                    dt = 0.9
                    dB = np.sqrt(dt) * np.random.normal(size=(n-1,))
                    #brownian motion starts at zero
                    B0 = np.zeros(shape=(1,))
                    #brownian motion is to concatenate the intial value with the cumulative sum of the increments.
                    B = np.concatenate((B0, np.cumsum(dB)))
                    brownian = (B * 255).astype(np.uint8)
                    brownian_noise = brownian.reshape(h,w,3)
                    im = cv2.add(img, brownian_noise)
                    print('brownian var 0.9')
                elif ('var0.09' in noise_type):
                    dt = 0.09
                    dB = np.sqrt(dt) * np.random.normal(size=(n-1,))
                    B0 = np.zeros(shape=(1,))
                    B = np.concatenate((B0, np.cumsum(dB)))
                    brownian = (B * 255).astype(np.uint8)
                    brownian_noise = brownian.reshape(h,w,3)
                    im = cv2.add(img, brownian_noise)
                    print('brownian var 0.09')
                elif ('var0.009' in noise_type):
                    dt = 0.009
                    dB = np.sqrt(dt) * np.random.normal(size=(n-1,))
                    B0 = np.zeros(shape=(1,))
                    B = np.concatenate((B0, np.cumsum(dB)))
                    brownian = (B * 255).astype(np.uint8)
                    brownian_noise = brownian.reshape(h,w,3)
                    im = cv2.add(img, brownian_noise)
                    print('brownian var 0.009')
            print("Brownian")
        return im

    def add_periodic_noise(noise_type):
        if ('periodic' in noise_type):
            h, w = img.shape[:2]
            size = img.size
            if ('periodic_wavelet' in noise_type):
                if ('var3.14' in noise_type):
                    time = (np.linspace(-np.pi, np.pi, size))
                    amplitude = np.sin(time)
                    periodic_array = (amplitude * 255).astype(np.uint8)
                    periodic_noise = periodic_array.reshape(h,w,3)
                    im_noise = cv2.add(img, periodic_noise)
                    im_bayes = denoise_wavelet(im_noise, method='BayesShrink', mode='soft',
                                        wavelet='bior1.5',
                                        multichannel= True, convert2ycbcr=True)
                    data = (255 * im_bayes)
                    im = data.astype(np.uint8)
                    print('periodic wavelet amplitude pi')
                elif ('var100' in noise_type):
                    time = (np.linspace(-100, 100, size))
                    amplitude = np.sin(time)
                    periodic_array = (amplitude * 255).astype(np.uint8)
                    periodic_noise = periodic_array.reshape(h,w,3)
                    im_noise = cv2.add(img, periodic_noise)
                    im_bayes = denoise_wavelet(im_noise, method='BayesShrink', mode='soft',
                                        wavelet='bior1.5',
                                        multichannel= True, convert2ycbcr=True)
                    data = (255 * im_bayes)
                    im = data.astype(np.uint8)
                    print('periodic wavelet amplitude 100')
                elif ('varsize' in noise_type):
                    time = (np.linspace(-size, size, size))
                    amplitude = np.sin(time)
                    periodic_array = (amplitude * 255).astype(np.uint8)
                    periodic_noise = periodic_array.reshape(h,w,3)
                    im_noise = cv2.add(img, periodic_noise)
                    im_bayes = denoise_wavelet(im_noise, method='BayesShrink', mode='soft',
                                        wavelet='bior1.5',
                                        multichannel= True, convert2ycbcr=True)
                    data = (255 * im_bayes)
                    im = data.astype(np.uint8)
                    print('periodic wavelet amplitude size')
            elif('periodic_gaus_blur' in noise_type):
                k_size = 3
                if ('var3.14' in noise_type):
                    time = (np.linspace(-np.pi, np.pi, size))
                    amplitude = np.sin(time)
                    periodic_array = (amplitude * 255).astype(np.uint8)
                    periodic_noise = periodic_array.reshape(h,w,3)
                    im_noise = cv2.add(img, periodic_noise)
                    im = cv2.GaussianBlur(im_noise, (k_size, k_size), 0)
                    print('periodic gausblur amplitude pi')
                elif ('var100' in noise_type):
                    time = (np.linspace(-100, 100, size))
                    amplitude = np.sin(time)
                    periodic_array = (amplitude * 255).astype(np.uint8)
                    periodic_noise = periodic_array.reshape(h,w,3)
                    im_noise = cv2.add(img, periodic_noise)
                    im = cv2.GaussianBlur(im_noise, (k_size, k_size), 0)
                    print('periodic gausblur amplitude 100')
                elif ('varsize' in noise_type):
                    time = (np.linspace(-size, size, size))
                    amplitude = np.sin(time)
                    periodic_array = (amplitude * 255).astype(np.uint8)
                    periodic_noise = periodic_array.reshape(h,w,3)
                    im_noise = cv2.add(img, periodic_noise)
                    im = cv2.GaussianBlur(im_noise, (k_size, k_size), 0)
                    print('periodic gausblur amplitude size')
            elif('periodic_mean' in noise_type):
                k_size = 3
                if ('var3.14' in noise_type):
                    time = (np.linspace(-np.pi, np.pi, size))
                    amplitude = np.sin(time)
                    periodic_array = (amplitude * 255).astype(np.uint8)
                    periodic_noise = periodic_array.reshape(h,w,3)
                    im_noise = cv2.add(img, periodic_noise)
                    im = cv2.blur(im_noise, (k_size, k_size))
                    print('periodic mean amplitude pi')
                elif ('var100' in noise_type):
                    time = (np.linspace(-100, 100, size))
                    amplitude = np.sin(time)
                    periodic_array = (amplitude * 255).astype(np.uint8)
                    periodic_noise = periodic_array.reshape(h,w,3)
                    im_noise = cv2.add(img, periodic_noise)
                    im = cv2.blur(im_noise, (k_size, k_size))
                    print('periodic mean amplitude 100')
                elif ('varsize' in noise_type):
                    time = (np.linspace(-size, size, size))
                    amplitude = np.sin(time)
                    periodic_array = (amplitude * 255).astype(np.uint8)
                    periodic_noise = periodic_array.reshape(h,w,3)
                    im_noise = cv2.add(img, periodic_noise)
                    im = cv2.blur(im_noise, (k_size, k_size))
                    print('periodic mean amplitude size')
            elif('periodic_median' in noise_type):
                k_size = 3
                if ('var3.14' in noise_type):
                    time = (np.linspace(-np.pi, np.pi, size))
                    amplitude = np.sin(time)
                    periodic_array = (amplitude * 255).astype(np.uint8)
                    periodic_noise = periodic_array.reshape(h,w,3)
                    im_noise = cv2.add(img, periodic_noise)
                    im = cv2.medianBlur(im_noise, k_size)
                    print('periodic median amplitude pi')
                elif ('var100' in noise_type):
                    time = (np.linspace(-100, 100, size))
                    amplitude = np.sin(time)
                    periodic_array = (amplitude * 255).astype(np.uint8)
                    periodic_noise = periodic_array.reshape(h,w,3)
                    im_noise = cv2.add(img, periodic_noise)
                    im = cv2.medianBlur(im_noise, k_size)
                    print('periodic median amplitude 100')
                elif ('varsize' in noise_type):
                    time = (np.linspace(-size, size, size))
                    amplitude = np.sin(time)
                    periodic_array = (amplitude * 255).astype(np.uint8)
                    periodic_noise = periodic_array.reshape(h,w,3)
                    im_noise = cv2.add(img, periodic_noise)
                    im = cv2.medianBlur(im_noise, k_size)
                    print('periodic median amplitude size')
            elif('periodic_bilateral' in noise_type):
                diameter = 9      #the diameter of each pixel in the neighborhood used during filtering
                sigmaColor = 20     #sigma of grey/color space.
                sigmaSpace = 100    #Large value means farther pixels influence each other.
                if ('var3.14' in noise_type):
                    time = (np.linspace(-np.pi, np.pi, size))
                    amplitude = np.sin(time)
                    periodic_array = (amplitude * 255).astype(np.uint8)
                    periodic_noise = periodic_array.reshape(h,w,3)
                    im_noise = cv2.add(img, periodic_noise)
                    im = cv2.bilateralFilter(im_noise, diameter, sigmaColor, sigmaSpace, borderType=cv2.BORDER_CONSTANT)
                    print('periodic bilateral amplitude pi')
                elif ('var100' in noise_type):
                    time = (np.linspace(-100, 100, size))
                    amplitude = np.sin(time)
                    periodic_array = (amplitude * 255).astype(np.uint8)
                    periodic_noise = periodic_array.reshape(h,w,3)
                    im_noise = cv2.add(img, periodic_noise)
                    im = cv2.bilateralFilter(im_noise, diameter, sigmaColor, sigmaSpace, borderType=cv2.BORDER_CONSTANT)
                    print('periodic bilateral amplitude 100')
                elif ('varsize' in noise_type):
                    time = (np.linspace(-size, size, size))
                    amplitude = np.sin(time)
                    periodic_array = (amplitude * 255).astype(np.uint8)
                    periodic_noise = periodic_array.reshape(h,w,3)
                    im_noise = cv2.add(img, periodic_noise)
                    im = cv2.bilateralFilter(im_noise, diameter, sigmaColor, sigmaSpace, borderType=cv2.BORDER_CONSTANT)
                    print('periodic bilateral amplitude size')
            else:
                if ('var3.14' in noise_type):
                    time = (np.linspace(-np.pi, np.pi, size))
                    amplitude = np.sin(time)
                    periodic_array = (amplitude * 255).astype(np.uint8)
                    periodic_noise = periodic_array.reshape(h,w,3)
                    im = cv2.add(img, periodic_noise)
                    print('periodic amplitude pi')
                elif ('var100' in noise_type):
                    time = (np.linspace(-100, 100, size))
                    amplitude = np.sin(time)
                    periodic_array = (amplitude * 255).astype(np.uint8)
                    periodic_noise = periodic_array.reshape(h,w,3)
                    im = cv2.add(img, periodic_noise)
                    print('periodic amplitude 100')
                elif ('varsize' in noise_type):
                    time = (np.linspace(-size, size, size))
                    amplitude = np.sin(time)
                    periodic_array = (amplitude * 255).astype(np.uint8)
                    periodic_noise = periodic_array.reshape(h,w,3)
                    im = cv2.add(img, periodic_noise)
                    print('periodic amplitude size')
            print("Periodic")
        return im

    def add_gamma_noise(noise_type):
        if ('gamma' in noise_type):
            image = img_as_float(img)
            a = 1.99
            if ('gamma_wavelet' in noise_type):
                if ('var0.05' in noise_type):
                    gamma_dist = gamma.rvs(a, loc=0., scale=0.05, size=image.shape)
                    gamma_array = cv2.add(image, gamma_dist)
                    im_bayes = denoise_wavelet(gamma_array, method='BayesShrink', mode='soft',
                                        wavelet='bior1.5',
                                        multichannel= True, convert2ycbcr=True)
                    data = (255 * im_bayes)
                    im = data.astype(np.uint8)
                    print('gamma wavelet var 0.05')
                elif ('var0.1' in noise_type):
                    gamma_dist = gamma.rvs(a, loc=0., scale=0.1, size=image.shape)
                    gamma_array = cv2.add(image, gamma_dist)
                    im_bayes = denoise_wavelet(gamma_array, method='BayesShrink', mode='soft',
                                        wavelet='bior1.5',
                                        multichannel= True, convert2ycbcr=True)
                    data = (255 * im_bayes)
                    im = data.astype(np.uint8)
                    print('gamma wavelet var 0.1')
                elif ('var0.2' in noise_type):
                    gamma_dist = gamma.rvs(a, loc=0., scale=0.2, size=image.shape)
                    gamma_array = cv2.add(image, gamma_dist)
                    im_bayes = denoise_wavelet(gamma_array, method='BayesShrink', mode='soft',
                                        wavelet='bior1.5',
                                        multichannel= True, convert2ycbcr=True)
                    data = (255 * im_bayes)
                    im = data.astype(np.uint8)
                    print('gamma wavelet var 0.2')
            elif('gamma_gaus_blur' in noise_type):
                size = 3
                if ('var0.05' in noise_type):
                    gamma_dist = gamma.rvs(a, loc=0., scale=0.05, size=image.shape)
                    gamma_array = cv2.add(image, gamma_dist)
                    im_noise = (gamma_array * 255).astype(np.uint8)
                    im = cv2.GaussianBlur(im_noise, (size, size), 0)
                    print('gamma gausblur var 0.05')
                elif ('var0.1' in noise_type):
                    gamma_dist = gamma.rvs(a, loc=0., scale=0.1, size=image.shape)
                    gamma_array = cv2.add(image, gamma_dist)
                    im_noise = (gamma_array * 255).astype(np.uint8)
                    im = cv2.GaussianBlur(im_noise, (size, size), 0)
                    print('gamma gausblur var 0.1')
                elif ('var0.2' in noise_type):
                    gamma_dist = gamma.rvs(a, loc=0., scale=0.2, size=image.shape)
                    gamma_array = cv2.add(image, gamma_dist)
                    im_noise = (gamma_array * 255).astype(np.uint8)
                    im = cv2.GaussianBlur(im_noise, (size, size), 0)
                    print('gamma gausblur var 0.2')
            elif('gamma_mean' in noise_type):
                size = 3
                if ('var0.05' in noise_type):
                    gamma_dist = gamma.rvs(a, loc=0., scale=0.05, size=image.shape)
                    gamma_array = cv2.add(image, gamma_dist)
                    im_noise = (gamma_array * 255).astype(np.uint8)
                    im = cv2.blur(im_noise, (size, size))
                    print('gamma mean var 0.05')
                elif ('var0.1' in noise_type):
                    gamma_dist = gamma.rvs(a, loc=0., scale=0.1, size=image.shape)
                    gamma_array = cv2.add(image, gamma_dist)
                    im_noise = (gamma_array * 255).astype(np.uint8)
                    im = cv2.blur(im_noise, (size, size))
                    print('gamma mean var 0.1')
                elif ('var0.2' in noise_type):
                    gamma_dist = gamma.rvs(a, loc=0., scale=0.2, size=image.shape)
                    gamma_array = cv2.add(image, gamma_dist)
                    im_noise = (gamma_array * 255).astype(np.uint8)
                    im = cv2.blur(im_noise, (size, size))
                    print('gamma mean var 0.2')
            elif('gamma_median' in noise_type):
                size = 3
                if ('var0.05' in noise_type):
                    gamma_dist = gamma.rvs(a, loc=0., scale=0.05, size=image.shape)
                    gamma_array = cv2.add(image, gamma_dist)
                    im_noise = (gamma_array * 255).astype(np.uint8)
                    im = cv2.medianBlur(im_noise, size)
                    print('gamma median var 0.05')
                elif ('var0.1' in noise_type):
                    gamma_dist = gamma.rvs(a, loc=0., scale=0.1, size=image.shape)
                    gamma_array = cv2.add(image, gamma_dist)
                    im_noise = (gamma_array * 255).astype(np.uint8)
                    im = cv2.medianBlur(im_noise, size)
                    print('gamma median var 0.1')
                elif ('var0.2' in noise_type):
                    gamma_dist = gamma.rvs(a, loc=0., scale=0.2, size=image.shape)
                    gamma_array = cv2.add(image, gamma_dist)
                    im_noise = (gamma_array * 255).astype(np.uint8)
                    im = cv2.medianBlur(im_noise, size)
                    print('gamma median var 0.2')
            elif('gamma_bilateral' in noise_type):
                diameter = 9      #the diameter of each pixel in the neighborhood used during filtering
                sigmaColor = 20     #sigma of grey/color space.
                sigmaSpace = 100    #Large value means farther pixels influence each other.
                if ('var0.05' in noise_type):
                    gamma_dist = gamma.rvs(a, loc=0., scale=0.05, size=image.shape)
                    gamma_array = cv2.add(image, gamma_dist)
                    im_noise = (gamma_array * 255).astype(np.uint8)
                    im = cv2.bilateralFilter(im_noise, diameter, sigmaColor, sigmaSpace, borderType=cv2.BORDER_CONSTANT)
                    print('gamma bilateral var 0.05')
                elif ('var0.1' in noise_type):
                    gamma_dist = gamma.rvs(a, loc=0., scale=0.1, size=image.shape)
                    gamma_array = cv2.add(image, gamma_dist)
                    im_noise = (gamma_array * 255).astype(np.uint8)
                    im = cv2.bilateralFilter(im_noise, diameter, sigmaColor, sigmaSpace, borderType=cv2.BORDER_CONSTANT)
                    print('gamma bilateral var 0.1')
                elif ('var0.2' in noise_type):
                    gamma_dist = gamma.rvs(a, loc=0., scale=0.2, size=image.shape)
                    gamma_array = cv2.add(image, gamma_dist)
                    im_noise = (gamma_array * 255).astype(np.uint8)
                    im = cv2.bilateralFilter(im_noise, diameter, sigmaColor, sigmaSpace, borderType=cv2.BORDER_CONSTANT)
                    print('gamma bilateral var 0.2')
            else:
                if ('var0.05' in noise_type):
                    #gamma_dist = gamma.rvs(a, loc=0., scale=0.1, size=image.shape)
                    gamma_dist = gamma.rvs(a, loc=0., scale=0.05, size=image.shape)
                    gamma_array = cv2.add(image, gamma_dist)
                    #im = gamma_array
                    im = (gamma_array * 255).astype(np.uint8)
                    print('gamma var 0.05')
                elif ('var0.1' in noise_type):
                    #gamma_dist = gamma.rvs(a, loc=0., scale=0.3, size=image.shape)
                    gamma_dist = gamma.rvs(a, loc=0., scale=0.1, size=image.shape)
                    gamma_array = cv2.add(image, gamma_dist)
                    #im = gamma_array
                    im = (gamma_array * 255).astype(np.uint8)
                    print('gamma var 0.1')
                elif ('var0.2' in noise_type):
                    #gamma_dist = gamma.rvs(a, loc=0., scale=0.7, size=image.shape)
                    gamma_dist = gamma.rvs(a, loc=0., scale=0.2, size=image.shape)
                    gamma_array = cv2.add(image, gamma_dist)
                    #im = gamma_array
                    im = (gamma_array * 255).astype(np.uint8)
                    print('gamma var 0.2')
            print("Gamma")
        return im

    def add_rayleigh_noise(noise_type):
        if ('rayleigh' in noise_type):
            image = img_as_float(img)
            if ('rayleigh_wavelet' in noise_type):
                if ('var0.1' in noise_type):
                    rayleigh_dist = rayleigh.rvs(loc=0., scale=0.1, size=image.shape)
                    rayleigh_array = cv2.add(image, rayleigh_dist)
                    im_bayes = denoise_wavelet(rayleigh_array, method='BayesShrink', mode='soft',
                                        wavelet='bior1.5',
                                        multichannel= True, convert2ycbcr=True)
                    data = (255 * im_bayes)
                    im = data.astype(np.uint8)
                    print('rayleigh wavelet var 0.1')
                elif ('var0.2' in noise_type):
                    rayleigh_dist = rayleigh.rvs(loc=0., scale=0.2, size=image.shape)
                    rayleigh_array = cv2.add(image, rayleigh_dist)
                    im_bayes = denoise_wavelet(rayleigh_array, method='BayesShrink', mode='soft',
                                        wavelet='bior1.5',
                                        multichannel= True, convert2ycbcr=True)
                    data = (255 * im_bayes)
                    im = data.astype(np.uint8)
                    print('rayleigh wavelet var 0.2')
                elif ('var0.3' in noise_type):
                    rayleigh_dist = rayleigh.rvs(loc=0., scale=0.3, size=image.shape)
                    rayleigh_array = cv2.add(image, rayleigh_dist)
                    im_bayes = denoise_wavelet(rayleigh_array, method='BayesShrink', mode='soft',
                                        wavelet='bior1.5',
                                        multichannel= True, convert2ycbcr=True)
                    data = (255 * im_bayes)
                    im = data.astype(np.uint8)
                    print('rayleigh wavelet var 0.3')
            elif('rayleigh_gaus_blur' in noise_type):
                size = 3
                if ('var0.1' in noise_type):
                    rayleigh_dist = rayleigh.rvs(loc=0., scale=0.1, size=image.shape)
                    rayleigh_array = cv2.add(image, rayleigh_dist)
                    im_noise = (rayleigh_array * 255).astype(np.uint8)
                    im = cv2.GaussianBlur(im_noise, (size, size), 0)
                    print('rayleigh gausblur 0.1')
                elif ('var0.2' in noise_type):
                    rayleigh_dist = rayleigh.rvs(loc=0., scale=0.2, size=image.shape)
                    rayleigh_array = cv2.add(image, rayleigh_dist)
                    im_noise = (rayleigh_array * 255).astype(np.uint8)
                    im = cv2.GaussianBlur(im_noise, (size, size), 0)
                    print('rayleigh gausblur 0.2')
                elif ('var0.3' in noise_type):
                    rayleigh_dist = rayleigh.rvs(loc=0., scale=0.3, size=image.shape)
                    rayleigh_array = cv2.add(image, rayleigh_dist)
                    im_noise = (rayleigh_array * 255).astype(np.uint8)
                    im = cv2.GaussianBlur(im_noise, (size, size), 0)
                    print('rayleigh gausblur 0.3')
            elif('rayleigh_mean' in noise_type):
                size = 3
                if ('var0.1' in noise_type):
                    rayleigh_dist = rayleigh.rvs(loc=0., scale=0.1, size=image.shape)
                    rayleigh_array = cv2.add(image, rayleigh_dist)
                    im_noise = (rayleigh_array * 255).astype(np.uint8)
                    im = cv2.blur(im_noise, (size, size))
                    print('rayleigh mean 0.1')
                elif ('var0.2' in noise_type):
                    rayleigh_dist = rayleigh.rvs(loc=0., scale=0.2, size=image.shape)
                    rayleigh_array = cv2.add(image, rayleigh_dist)
                    im_noise = (rayleigh_array * 255).astype(np.uint8)
                    im = cv2.blur(im_noise, (size, size))
                    print('rayleigh mean 0.2')
                elif ('var0.3' in noise_type):
                    rayleigh_dist = rayleigh.rvs(loc=0., scale=0.3, size=image.shape)
                    rayleigh_array = cv2.add(image, rayleigh_dist)
                    im_noise = (rayleigh_array * 255).astype(np.uint8)
                    im = cv2.blur(im_noise, (size, size))
                    print('rayleigh mean 0.3')
            elif('rayleigh_median' in noise_type):
                size = 3
                if ('var0.1' in noise_type):
                    rayleigh_dist = rayleigh.rvs(loc=0., scale=0.1, size=image.shape)
                    rayleigh_array = cv2.add(image, rayleigh_dist)
                    im_noise = (rayleigh_array * 255).astype(np.uint8)
                    im = cv2.medianBlur(im_noise, size)
                    print('rayleigh median 0.1')
                elif ('var0.2' in noise_type):
                    rayleigh_dist = rayleigh.rvs(loc=0., scale=0.2, size=image.shape)
                    rayleigh_array = cv2.add(image, rayleigh_dist)
                    im_noise = (rayleigh_array * 255).astype(np.uint8)
                    im = cv2.medianBlur(im_noise, size)
                    print('rayleigh median 0.2')
                elif ('var0.3' in noise_type):
                    rayleigh_dist = rayleigh.rvs(loc=0., scale=0.3, size=image.shape)
                    rayleigh_array = cv2.add(image, rayleigh_dist)
                    im_noise = (rayleigh_array * 255).astype(np.uint8)
                    im = cv2.medianBlur(im_noise, size)
                    print('rayleigh median 0.3')
            elif('rayleigh_bilateral' in noise_type):
                diameter = 9      #the diameter of each pixel in the neighborhood used during filtering
                sigmaColor = 20     #sigma of grey/color space.
                sigmaSpace = 100    #Large value means farther pixels influence each other.
                if ('var0.1' in noise_type):
                    rayleigh_dist = rayleigh.rvs(loc=0., scale=0.1, size=image.shape)
                    rayleigh_array = cv2.add(image, rayleigh_dist)
                    im_noise = (rayleigh_array * 255).astype(np.uint8)
                    im = cv2.bilateralFilter(im_noise, diameter, sigmaColor, sigmaSpace, borderType=cv2.BORDER_CONSTANT)
                    print('rayleigh bilateral 0.1')
                elif ('var0.2' in noise_type):
                    rayleigh_dist = rayleigh.rvs(loc=0., scale=0.2, size=image.shape)
                    rayleigh_array = cv2.add(image, rayleigh_dist)
                    im_noise = (rayleigh_array * 255).astype(np.uint8)
                    im = cv2.bilateralFilter(im_noise, diameter, sigmaColor, sigmaSpace, borderType=cv2.BORDER_CONSTANT)
                    print('rayleigh bilateral 0.2')
                elif ('var0.3' in noise_type):
                    rayleigh_dist = rayleigh.rvs(loc=0., scale=0.3, size=image.shape)
                    rayleigh_array = cv2.add(image, rayleigh_dist)
                    im_noise = (rayleigh_array * 255).astype(np.uint8)
                    im = cv2.bilateralFilter(im_noise, diameter, sigmaColor, sigmaSpace, borderType=cv2.BORDER_CONSTANT)
                    print('rayleigh bilateral 0.3')
            else:
                if ('var0.1' in noise_type):
                    rayleigh_dist = rayleigh.rvs(loc=0., scale=0.1, size=image.shape)
                    rayleigh_array = cv2.add(image, rayleigh_dist)
                    #im = rayleigh_array
                    im = (rayleigh_array * 255).astype(np.uint8)
                    print('rayleigh var 0.1')
                elif ('var0.2' in noise_type):
                    rayleigh_dist = rayleigh.rvs(loc=0., scale=0.2, size=image.shape)
                    rayleigh_array = cv2.add(image, rayleigh_dist)
                    #im = rayleigh_array
                    im = (rayleigh_array * 255).astype(np.uint8)
                    print('rayleigh var 0.2')
                elif ('var0.3' in noise_type):
                    rayleigh_dist = rayleigh.rvs(loc=0., scale=0.3, size=image.shape)
                    rayleigh_array = cv2.add(image, rayleigh_dist)                    
                    #im = rayleigh_array
                    im = (rayleigh_array * 255).astype(np.uint8)
                    print('rayleigh var 0.3')
            print("Rayleigh")
        return im
        #return rayleigh_array
    """def add_bloom():
        img = cv2.imread(imdb.image_path_at(i))
        image = img_as_float(img)
        in_im = np.zeros(img.shape[:2]).astype(np.float64)
        #set illumination point
        in_im[:1, :1] = 1
        #init Lightsource
        ls = matplotlib.colors.LightSource(azdeg=315, altdeg=45, hsv_min_val=0, hsv_max_val=1, hsv_min_sat=1, hsv_max_sat=0)
        #claculate illumination intensity
        intensity = ls.hillshade(in_im, vert_exag=2)
        _intensity = intensity[:, :, newaxis]
        #combine rgb image with intensity
        im = ls.blend_overlay(rgb=image, intensity=_intensity)
        im = (im * 255).astype(np.uint8)
        print('bloom')
        return im"""
    def add_bloom():
        img = cv2.imread(imdb.image_path_at(i))
        im = am.add_sun_flare(img, flare_center=(100,100), angle=-math.pi/4)
        print("bloom effect")
        return im
    def add_shader():
        factor = 3
        im = Image.open(imdb.image_path_at(i))
        im_out = ImageEnhance.Brightness(im).enhance(factor)
        im = np.array(im_out)
        print('PIL enhance')
        return im
    def retain_original():
        #im = cv2.imread(roidb[i]['image'])
        im = cv2.imread(imdb.image_path_at(i))
        #introduce gaussian noise.
        print('original')
        return im


    
    #if ('mix' in noise):
    if ('mix' in noise):
        if ('var_low' in noise):
            noise_list = ['gaussian_var0.1', 'poisson', 'speckle_var0.5',
                            'sap_var0.2', 'uniform_var0.2', 'gamma_var0.05',
                            'rayleigh_var0.1','periodic_var3.14','brownian_var0.9', 'quant_var3', 'original', 'bloom', 'shader']
            noise_type = random.choice(noise_list)
            print(noise_type)
        elif ('var_medium' in noise):
            noise_list = ['gaussian_var1.0', 'poisson', 'speckle_var1.0',
                            'sap_var0.4', 'uniform_var0.6', 'gamma_var0.1',
                            'rayleigh_var0.2','periodic_var100','brownian_var0.09', 'quant_var7', 'original', 'shader', 'bloom']
            noise_type = random.choice(noise_list)
        elif ('var_high' in noise):
            noise_list = ['gaussian_var1.5', 'poisson', 'speckle_var2.0',  
                            'sap_var0.8', 'uniform_var1.2', 'gamma_var0.2', 
                            'rayleigh_var0.3','periodic_varsize','brownian_var0.009', 'quant_var10', 'original', 'shader', 'bloom']
            noise_type = random.choice(noise_list)
        elif ('var_all' in noise):
            noise_list = ['gaussian_var0.1', 'poisson', 'speckle_var0.5',
                            'sap_var0.2', 'uniform_var0.2', 'gamma_var0.05', 'gamma_var0.05', 'rayleigh_var0.2',
                            'rayleigh_var0.1','periodic_var3.14','brownian_var0.9', 'quant_var3', 'gamma_var0.1', 'rayleigh_var0.1',
                            'gaussian_var1.0', 'poisson', 'speckle_var1.0',
                            'sap_var0.4', 'uniform_var0.6', 'gamma_var0.1', 'shader', 'original', 'shader', 'bloom',
                            'rayleigh_var0.2','periodic_var100','brownian_var0.09', 'quant_var7', 
                            'gaussian_var1.5', 'poisson', 'speckle_var2.0',
                            'sap_var0.8', 'uniform_var1.2', 'gamma_var0.2', 'shader', 'original', 
                            'rayleigh_var0.3','periodic_varsize','brownian_var0.009', 'quant_var10', 'original', 'shader']
            noise_type = random.choice(noise_list)
        if ('gaussian' in noise_type):
            im = add_gaussian_noise(noise_type)
            #im = retain_original()
        elif('poisson' in noise_type):
            im = add_poisson_noise(noise_type)
            #im = retain_original()
        elif('sap' in noise_type):
            im = add_sap_noise(noise_type)
            #im = retain_original()
        elif('speckle' in noise_type):
            im = add_speckle_noise(noise_type)
            #im = retain_original()
        elif('periodic' in noise_type):
            im = add_periodic_noise(noise_type)
            #im = retain_original()
        elif('brownian' in noise_type):
            im = add_brownian_noise(noise_type)
            #im = retain_original()
        elif('quant' in noise_type):
            im = add_quant_noise(noise_type)
            #im = retain_original()
        elif('uniform' in noise_type):
            im = add_uniform_noise(noise_type)
            #im = retain_original()
        elif('gamma' in noise_type):
            im = add_gamma_noise(noise_type)
            #im = retain_original()
        elif('rayleigh' in noise_type):
            im = add_rayleigh_noise(noise_type)
            #im = retain_original()
        elif('bloom' in noise_type):
            im = add_bloom()
            #im = retain_original()
        elif('shader' in noise_type):
            im = add_shader()
            #im = retain_original()
        else :
            im = retain_original()
    elif ('gaussian' in noise):
        noise_list = ['gaussian_var0.1', 'gaussian_var1.0', 'gaussian_var1.5']
        noise_type = random.choice(noise_list)
        #noise_type = noise
        im = add_gaussian_noise(noise_type)
        #im = add_gaussian_noise(noise)
        #im = retain_original()
    elif('poisson' in noise):
        #noise_type = noise
        noise_type = 'poisson'
        im = add_poisson_noise(noise_type)
        #im = add_poisson_noise(noise)
        #im = retain_original()
    elif('sap' in noise):
        #noise_list = ['sap_var0.2', 'sap_var0.4', 'sap_var0.8']
        #noise_type = random.choice(noise_list)
        #noise_type = noise
        #im = add_sap_noise(noise_type)
        #im = add_sap_noise(noise)
        im = retain_original()
    elif('speckle' in noise):
        #noise_list = ['speckle_var0.5', 'speckle_var1.0', 'speckle_var2.0']
        #noise_type = random.choice(noise_list)
        noise_type = noise
        im = add_speckle_noise(noise_type)
        #im = add_speckle_noise(noise)
        #im = retain_original()
    elif('periodic' in noise):
        #noise_list = ['periodic_var3.14', 'periodic_var100', 'periodic_varsize']
        #noise_type = random.choice(noise_list)
        noise_type = noise
        im = add_periodic_noise(noise_type)
        #im = add_periodic_noise(noise)
        #im = retain_original()
    elif('brownian' in noise):
        #noise_list = ['brownian_var0.9', 'brownian_var0.09', 'brownian_var0.009']
        #noise_type = random.choice(noise_list)
        noise_type = noise
        im = add_brownian_noise(noise_type)
        #im = add_brownian_noise(noise)
        #im = retain_original()
    elif('quant' in noise):
        #noise_list = ['quant_var3', 'quant_var7', 'quant_var10']
        #noise_type = random.choice(noise_list)
        #noise_type = noise
        #im = add_quant_noise(noise_type)
        #im = add_quant_noise(noise)
        im = retain_original()
    elif('uniform' in noise):
        #noise_list = ['uniform_var0.2', 'uniform_var0.6', 'uniform_var1.2']
        #noise_type = random.choice(noise_list)
        noise_type = noise
        im = add_uniform_noise(noise_type)
        #im = add_uniform_noise(noise)
        #im = retain_original()
    elif('gamma' in noise):
        #noise_list = ['gamma_var0.1', 'gamma_var0.3', 'gamma_var0.7']
        #noise_type = random.choice(noise_list)
        noise_type = noise
        im = add_gamma_noise(noise_type)
        #im = add_gamma_noise(noise)
        #im = retain_original()
    elif('rayleigh' in noise):
        #noise_list = ['rayleigh_var0.1', 'rayleigh_var0.2', 'rayleigh_var0.3']
        #noise_type = random.choice(noise_list)
        noise_type = noise
        im = add_rayleigh_noise(noise_type)
        #im = add_rayleigh_noise(noise)
        #im = retain_original()
    elif('bloom' in noise):
        #noise_type = noise
        im = add_bloom()
        #im = retain_original()
    elif('shader' in noise):
        #noise_type = noise
        im = add_shader()
        #im = retain_original()
    elif('curvelet' in noise):
        print('curvelet')    
    else :
        #noise_list = ['gaussian_var0.1', 'gaussian_var1.0', 'gaussian_var1.5']
        noise_type = 'gaussian_var0.1'
        #noise_type = random.choice(noise_list)
        im = add_gaussian_noise(noise_type)
        #im = add_gaussian_noise(gaussian_var0.1)
        k_size = 3
        #im = cv2.GaussianBlur(im, (k_size, k_size), 0)
        #print('gaussian blur mix')
        
        im = cv2.blur(im, (k_size, k_size))
        print('mean mix')
        #im = cv2.medianBlur(im, k_size)
        #print('median mix')
        #im_bayes = denoise_wavelet(im, method='BayesShrink', mode='soft', wavelet='bior1.5',
        #                            multichannel= True, convert2ycbcr=True)
        #data = (255 * im_bayes)
        #im = data.astype(np.uint8)
        #print('wavelet mix')

        #diameter = 9      #the diameter of each pixel in the neighborhood used during filtering
        #sigmaColor = 20     #sigma of grey/color space.
        #sigmaSpace = 100    #Large value means farther pixels influence each other.
        #im = cv2.bilateralFilter(im, diameter, sigmaColor, sigmaSpace, borderType=cv2.BORDER_CONSTANT)
        #print('bilateral mix')


        #im = retain_original()


    if ('gaus_blur' in noise):
        #k_size = 3
        #im = cv2.GaussianBlur(im, (k_size, k_size), 0)
        print('gaussian blur mix')
        #im = retain_original()
    elif ('mean' in noise):
        #k_size = 3
        #im = cv2.blur(im, (k_size, k_size))
        print('mean mix')
        #im = retain_original()
    elif ('median' in noise):
        #k_size = 3
        #im = cv2.medianBlur(im, k_size)
        print('median mix')
        #im = retain_original()
    elif ('wavelet' in noise):
        #im = img_as_float(im)
        #im_bayes = denoise_wavelet(im, method='BayesShrink', mode='soft',
        #                            wavelet_levels=3,
        #                            multichannel= True, convert2ycbcr=True)
        im_bayes = denoise_wavelet(im, method='BayesShrink', mode='soft', wavelet='bior1.5', 
                                    multichannel= True, convert2ycbcr=True)
        data = (255 * im_bayes)
        im = data.astype(np.uint8)
        print('wavelet mix')
        #im = retain_original()
    elif ('bilateral' in noise):
        #diameter = 9      #the diameter of each pixel in the neighborhood used during filtering
        #sigmaColor = 20     #sigma of grey/color space.
        #sigmaSpace = 100    #Large value means farther pixels influence each other.
        #im = cv2.bilateralFilter(im, diameter, sigmaColor, sigmaSpace, borderType=cv2.BORDER_CONSTANT)
        print('bilateral mix')
        #im = retain_original()
    elif('curvelet' in noise):
        img_path = imdb.image_path_at(i)
        #noise_list = ['gaussian_var1.0', 'poisson', 'speckle_var1.0',
        #                    'sap_var0.4', 'uniform_var0.6']
        noise_list = ['gaussian_var1.0', 'poisson', 'speckle_var1.0',
                            'sap_var0.4', 'uniform_var0.6', 'gamma_var0.3',
                            'rayleigh_var0.2','periodic_var100','brownian_var0.09', 'quant_var7', 'original', 'shader']
        noise_type = random.choice(noise_list)
        #img = os.system('python3 fdct.py ' )
        img = subprocess.check_output(["python3", "/home/mahesh/thesis/de-noise/tf-faster-rcnn/lib/model/fdct.py", noise_type, img_path])
        im = cv2.imread("temp.png")
        im = retain_original()
    


    _t['im_detect'].tic()
    scores, boxes = im_detect(sess, net, im)
    _t['im_detect'].toc()

    """with test_summary_writer.as_default():
        tf.summary.scalar('scores', scores)
        #tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)"""
    #testWriter = tf.FileWriter()
    #test_summary_writer.add_summary(scores, float(i))
    #print(scores)
    #print((scores.shape))
    #test_val = net.get_summary(sess, im)
    #test_summary_writer.add_summary(test_val, float(i))

    _t['misc'].tic()
    #ouput1 = net.model_summary(sess, im, train_op=False)

    # skip j = 0, because it's the background class
    #for j, cls in range(enumerate(imdb.classes[1:])):
    for j in range(1, imdb.num_classes):
      inds = np.where(scores[:, j] > thresh)[0]
      cls_scores = scores[inds, j]
      cls_boxes = boxes[inds, j*4:(j+1)*4]
      cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
        .astype(np.float32, copy=False)
      keep = nms(cls_dets, cfg.TEST.NMS)
      cls_dets = cls_dets[keep, :]
      all_boxes[j][i] = cls_dets
      #vis_detections(im, cls, cls_dets, thresh=0.8)
      inds = np.where(cls_dets[:, -1] >= 0.8)[0]
      avgscore = 0
      for ind in inds:
          score = cls_dets[ind, -1]
          #print(score)
          #print(type(score))
          #print(cls_dets[ind])
          summary = tf.Summary(value=[
              tf.Summary.Value(tag="score", simple_value=score),
              ])
          test_summary_writer.add_summary(summary, float(i))
          #wandb.tensorflow.log(summary)
          #wandb.tensorflow.log(tf.summary.merge_all())
          avgscore = score + avgscore 
          #wandb.log({"score": score, "noise type": noise, "iteration": i, "noise_variance": "high", "infer_on": "original"})
      """if (len(inds) == 0):
          avgscore = 0
          wandb.log({"score": avgscore, "noise type": noise, "iteration": i})
      elif (len(inds) >  0):
          avgscore = avgscore / len(inds) 
          wandb.log({"score": avgscore, "noise type": noise, "iteration": i})"""




    # Limit to max_per_image detections *over all classes*
    if max_per_image > 0:
      image_scores = np.hstack([all_boxes[j][i][:, -1]
                    for j in range(1, imdb.num_classes)])
      if len(image_scores) > max_per_image:
        image_thresh = np.sort(image_scores)[-max_per_image]
        for j in range(1, imdb.num_classes):
          keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
          all_boxes[j][i] = all_boxes[j][i][keep, :]

    _t['misc'].toc()

    #test_summary_writer.add_summary(scores, float(i))

    print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
        .format(i + 1, num_images, _t['im_detect'].average_time,
            _t['misc'].average_time))

    
    blobs, im_scales = _get_blobs(im)
    im_blob = blobs['data']
    blobs['im_info'] = np.array([im_blob.shape[1], im_blob.shape[2], im_scales[0]], dtype=np.float32)
    feed_dict = {net._image: blobs['data'], net._im_info: blobs['im_info']}
    graph = tf.get_default_graph()
    #output = graph.get_tensor_by_name('resnet_v1_101_2/block2/unit_1/bottleneck_v1/conv1/Conv2D:0')
    #feat1_ = sess.run(output, feed_dict=feed_dict)
    #feat11_ = feat1_.view().reshape(feat1_.shape[0], -1)
    #print(feat11_)
    #with tf.device("/gpu:1"):
    #    graph = tf.get_default_graph()
    """if i == 0:
        output0 = graph.get_tensor_by_name('resnet_v1_101_1/block1/unit_1/bottleneck_v1/conv3/Conv2D:0')
        feat0 = sess.run(output0, feed_dict=feed_dict)
        feat00 = feat0.view().reshape(feat0.shape[0], -1)
        output1 = graph.get_tensor_by_name('resnet_v1_101_2/block2/unit_1/bottleneck_v1/conv3/Conv2D:0')
        feat1 = sess.run(output1, feed_dict=feed_dict)
        feat11 = feat1.view().reshape(feat1.shape[0], -1)
        output2 = graph.get_tensor_by_name('resnet_v1_101_2/block2/unit_4/bottleneck_v1/conv3/Conv2D:0')
        feat2 = sess.run(output2, feed_dict=feed_dict)
        feat22 = feat2.view().reshape(feat2.shape[0], -1)
        output3 = graph.get_tensor_by_name('resnet_v1_101_2/block3/unit_8/bottleneck_v1/conv3/Conv2D:0')
        feat3 = sess.run(output3, feed_dict=feed_dict)
        feat33 = feat3.view().reshape(feat3.shape[0], -1)
        output4 = graph.get_tensor_by_name('resnet_v1_101_2/block3/unit_11/bottleneck_v1/conv3/Conv2D:0')
        feat4 = sess.run(output4, feed_dict=feed_dict)
        feat44 = feat4.view().reshape(feat4.shape[0], -1)
        output5 = graph.get_tensor_by_name('resnet_v1_101_2/block3/unit_14/bottleneck_v1/conv3/Conv2D:0')
        feat5 = sess.run(output5, feed_dict=feed_dict)
        feat55 = feat5.view().reshape(feat5.shape[0], -1)
        output6 = graph.get_tensor_by_name('resnet_v1_101_2/block3/unit_17/bottleneck_v1/conv3/Conv2D:0')
        feat6 = sess.run(output6, feed_dict=feed_dict)
        feat66 = feat6.view().reshape(feat6.shape[0], -1)
        output7 = graph.get_tensor_by_name('resnet_v1_101_2/block3/unit_20/bottleneck_v1/conv3/Conv2D:0')
        feat7 = sess.run(output7, feed_dict=feed_dict)
        feat77 = feat7.view().reshape(feat7.shape[0], -1)
        output8 = graph.get_tensor_by_name('resnet_v1_101_2/block3/unit_23/bottleneck_v1/conv3/Conv2D:0')
        feat8 = sess.run(output8, feed_dict=feed_dict)
        feat88 = feat8.view().reshape(feat8.shape[0], -1)


        #print("output shape of tensor out 0: ", output0.shape)
        #print("output shape of tensor feat0: ", feat0.shape )
        #print("output shape of tensor feat00: ", feat00.shape )

        #for res50
        output0 = graph.get_tensor_by_name('resnet_v1_50_1/block1/unit_1/bottleneck_v1/conv3/Conv2D:0')
        feat0 = sess.run(output0, feed_dict=feed_dict)
        feat00 = feat0.view().reshape(feat0.shape[0], -1)
        output1 = graph.get_tensor_by_name('resnet_v1_50_1/block1/unit_3/bottleneck_v1/conv3/Conv2D:0')
        feat1 = sess.run(output1, feed_dict=feed_dict)
        feat11 = feat1.view().reshape(feat1.shape[0], -1)
        output2 = graph.get_tensor_by_name('resnet_v1_50_2/block2/unit_1/bottleneck_v1/conv3/Conv2D:0')
        feat2 = sess.run(output2, feed_dict=feed_dict)
        feat22 = feat2.view().reshape(feat2.shape[0], -1)
        output3 = graph.get_tensor_by_name('resnet_v1_50_2/block2/unit_2/bottleneck_v1/conv3/Conv2D:0')
        feat3 = sess.run(output3, feed_dict=feed_dict)
        feat33 = feat3.view().reshape(feat3.shape[0], -1)
        output4 = graph.get_tensor_by_name('resnet_v1_50_2/block2/unit_3/bottleneck_v1/conv3/Conv2D:0')
        feat4 = sess.run(output4, feed_dict=feed_dict)
        feat44 = feat4.view().reshape(feat4.shape[0], -1)
        output5 = graph.get_tensor_by_name('resnet_v1_50_2/block2/unit_4/bottleneck_v1/conv3/Conv2D:0')
        feat5 = sess.run(output5, feed_dict=feed_dict)
        feat55 = feat5.view().reshape(feat5.shape[0], -1)
        output6 = graph.get_tensor_by_name('resnet_v1_50_2/block3/unit_3/bottleneck_v1/conv3/Conv2D:0')
        feat6 = sess.run(output6, feed_dict=feed_dict)
        feat66 = feat6.view().reshape(feat6.shape[0], -1)
        output7 = graph.get_tensor_by_name('resnet_v1_50_2/block3/unit_4/bottleneck_v1/conv3/Conv2D:0')
        feat7 = sess.run(output7, feed_dict=feed_dict)
        feat77 = feat7.view().reshape(feat7.shape[0], -1)
        #output8 = graph.get_tensor_by_name('resnet_v1_50_2/block3/unit_6/bottleneck_v1/conv3/Conv2D:0')
        #feat8 = sess.run(output8, feed_dict=feed_dict)
        #feat88 = feat8.view().reshape(feat8.shape[0], -1)

        #for VGG
        output0 = graph.get_tensor_by_name('vgg_16/conv1/conv1_1/Conv2D:0')
        feat0 = sess.run(output0, feed_dict=feed_dict)
        feat00 = feat0.view().reshape(feat0.shape[0], -1)
        output1 = graph.get_tensor_by_name('vgg_16/conv1/conv1_2/Conv2D:0')
        feat1 = sess.run(output1, feed_dict=feed_dict)
        feat11 = feat1.view().reshape(feat1.shape[0], -1)
        output2 = graph.get_tensor_by_name('vgg_16/conv2/conv2_2/Conv2D:0')
        feat2 = sess.run(output2, feed_dict=feed_dict)
        feat22 = feat2.view().reshape(feat2.shape[0], -1)
        #output3 = graph.get_tensor_by_name('vgg_16/conv3/conv3_2/Conv2D:0')
        #feat3 = sess.run(output3, feed_dict=feed_dict)
        #feat33 = feat3.view().reshape(feat3.shape[0], -1)
        output4 = graph.get_tensor_by_name('vgg_16/conv3/conv3_3/Conv2D:0')
        feat4 = sess.run(output4, feed_dict=feed_dict)
        feat44 = feat4.view().reshape(feat4.shape[0], -1)
        #output5 = graph.get_tensor_by_name('vgg_16/conv4/conv4_2/Conv2D:0')
        #feat5 = sess.run(output5, feed_dict=feed_dict)
        #feat55 = feat5.view().reshape(feat5.shape[0], -1)
        output6 = graph.get_tensor_by_name('vgg_16/conv4/conv4_3/Conv2D:0')
        feat6 = sess.run(output6, feed_dict=feed_dict)
        feat66 = feat6.view().reshape(feat6.shape[0], -1)
        #output7 = graph.get_tensor_by_name('vgg_16/conv5/conv5_2/Conv2D:0')
        #feat7 = sess.run(output7, feed_dict=feed_dict)
        #feat77 = feat7.view().reshape(feat7.shape[0], -1)
        output8 = graph.get_tensor_by_name('vgg_16/conv5/conv5_3/Conv2D:0')
        feat8 = sess.run(output8, feed_dict=feed_dict)
        feat88 = feat8.view().reshape(feat8.shape[0], -1)

        #output9 = graph.get_tensor_by_name('resnet_v1_101_2/block3/unit_23/bottleneck_v1/conv3/Conv2D:0')
        #feat9 = sess.run(output9, feed_dict=feed_dict)
        #feat99 = feat9.view().reshape(feat9.shape[0], -1)
        #output9 = graph.get_tensor_by_name('resnet_v1_101_3/rpn_conv/3x3/Conv2D:0')
        #feat9 = sess.run(output9, feed_dict=feed_dict)
        #feat99 = feat9.view().reshape(feat9.shape[0], -1)
        #output10 = graph.get_tensor_by_name('resnet_v1_101_4/block4/unit_1/bottleneck_v1/conv3/Conv2D:0')
        #feat10 = sess.run(output10, feed_dict=feed_dict)
        #feat100 = feat10.view().reshape(feat10.shape[0], -1)
        #output11 = graph.get_tensor_by_name('resnet_v1_101_4/block4/unit_3/bottleneck_v1/conv3/Conv2D:0')
        #feat11_ = sess.run(output11, feed_dict=feed_dict)
        #feat111 = feat11_.view().reshape(feat11_.shape[0], -1)
    else:
        output0 = graph.get_tensor_by_name('resnet_v1_101_1/block1/unit_1/bottleneck_v1/conv3/Conv2D:0')
        feat0 = sess.run(output0, feed_dict=feed_dict)
        output1 = graph.get_tensor_by_name('resnet_v1_101_2/block2/unit_1/bottleneck_v1/conv3/Conv2D:0')
        feat1 = sess.run(output1, feed_dict=feed_dict)
        #feat11 = tf.concat([feat11, feat1.view().reshape(feat1.shape[0], -1)],0)
        #feat11 = np.concatenate([feat11, feat1.view().reshape(feat1.shape[0], -1)],0)
        output2 = graph.get_tensor_by_name('resnet_v1_101_2/block2/unit_4/bottleneck_v1/conv3/Conv2D:0')
        feat2 = sess.run(output2, feed_dict=feed_dict)
        #feat22 = tf.concat([feat22, feat2.view().reshape(feat2.shape[0], -1)],0)
        #feat22 = np.concatenate([feat22, feat2.view().reshape(feat2.shape[0], -1)],0)
        output3 = graph.get_tensor_by_name('resnet_v1_101_2/block3/unit_8/bottleneck_v1/conv3/Conv2D:0')
        feat3 = sess.run(output3, feed_dict=feed_dict)
        #feat33 = tf.concat([feat33, feat3.view().reshape(feat3.shape[0], -1)],0)
        #feat33 = np.concatenate([feat33, feat3.view().reshape(feat3.shape[0], -1)],0)
        output4 = graph.get_tensor_by_name('resnet_v1_101_2/block3/unit_11/bottleneck_v1/conv3/Conv2D:0')
        feat4 = sess.run(output4, feed_dict=feed_dict)
        #feat44 = tf.concat([feat44, feat4.view().reshape(feat4.shape[0], -1)],0)
        #feat44 = np.concatenate([feat44, feat4.view().reshape(feat4.shape[0], -1)],0)
        output5 = graph.get_tensor_by_name('resnet_v1_101_2/block3/unit_14/bottleneck_v1/conv3/Conv2D:0')
        feat5 = sess.run(output5, feed_dict=feed_dict)
        #feat55 = tf.concat([feat55, feat5.view().reshape(feat5.shape[0], -1)],0)
        #feat55 = np.concatenate([feat55, feat5.view().reshape(feat5.shape[0], -1)],0)
        output6 = graph.get_tensor_by_name('resnet_v1_101_2/block3/unit_17/bottleneck_v1/conv3/Conv2D:0')
        feat6 = sess.run(output6, feed_dict=feed_dict)
        #feat66 = tf.concat([feat66, feat6.view().reshape(feat6.shape[0], -1)],0)
        #feat66 = np.concatenate([feat66, feat6.view().reshape(feat6.shape[0], -1)],0)
        output7 = graph.get_tensor_by_name('resnet_v1_101_2/block3/unit_20/bottleneck_v1/conv3/Conv2D:0')
        feat7 = sess.run(output7, feed_dict=feed_dict)
        #feat77 = tf.concat([feat77, feat7.view().reshape(feat7.shape[0], -1)],0)
        #feat77 = np.concatenate([feat77, feat7.view().reshape(feat7.shape[0], -1)],0)
        output8 = graph.get_tensor_by_name('resnet_v1_101_2/block3/unit_23/bottleneck_v1/conv3/Conv2D:0')
        feat8 = sess.run(output8, feed_dict=feed_dict)

        #print("output shape of tensor out 1: ", output1.shape)
        #print("output shape of tensor out 8:", output8.shape )

        #for res50
        output0 = graph.get_tensor_by_name('resnet_v1_50_1/block1/unit_1/bottleneck_v1/conv3/Conv2D:0')
        feat0 = sess.run(output0, feed_dict=feed_dict)
        output1 = graph.get_tensor_by_name('resnet_v1_50_1/block1/unit_3/bottleneck_v1/conv3/Conv2D:0')
        feat1 = sess.run(output1, feed_dict=feed_dict)
        output2 = graph.get_tensor_by_name('resnet_v1_50_2/block2/unit_1/bottleneck_v1/conv3/Conv2D:0')
        feat2 = sess.run(output2, feed_dict=feed_dict)
        output3 = graph.get_tensor_by_name('resnet_v1_50_2/block2/unit_2/bottleneck_v1/conv3/Conv2D:0')
        feat3 = sess.run(output3, feed_dict=feed_dict)
        output4 = graph.get_tensor_by_name('resnet_v1_50_2/block2/unit_3/bottleneck_v1/conv3/Conv2D:0')
        feat4 = sess.run(output4, feed_dict=feed_dict)
        output5 = graph.get_tensor_by_name('resnet_v1_50_2/block2/unit_4/bottleneck_v1/conv3/Conv2D:0')
        feat5 = sess.run(output5, feed_dict=feed_dict)
        output6 = graph.get_tensor_by_name('resnet_v1_50_2/block3/unit_3/bottleneck_v1/conv3/Conv2D:0')
        feat6 = sess.run(output6, feed_dict=feed_dict)
        output7 = graph.get_tensor_by_name('resnet_v1_50_2/block3/unit_4/bottleneck_v1/conv3/Conv2D:0')
        feat7 = sess.run(output7, feed_dict=feed_dict)
        #output8 = graph.get_tensor_by_name('resnet_v1_50_2/block3/unit_6/bottleneck_v1/conv3/Conv2D:0')
        #feat8 = sess.run(output8, feed_dict=feed_dict)


        #For VGG:
        output0 = graph.get_tensor_by_name('vgg_16/conv1/conv1_1/Conv2D:0')
        feat0 = sess.run(output0, feed_dict=feed_dict)
        output1 = graph.get_tensor_by_name('vgg_16/conv1/conv1_2/Conv2D:0')
        feat1 = sess.run(output1, feed_dict=feed_dict)
        output2 = graph.get_tensor_by_name('vgg_16/conv2/conv2_2/Conv2D:0')
        feat2 = sess.run(output2, feed_dict=feed_dict)
        #output3 = graph.get_tensor_by_name('vgg_16/conv3/conv3_2/Conv2D:0')
        #feat3 = sess.run(output3, feed_dict=feed_dict)
        output4 = graph.get_tensor_by_name('vgg_16/conv3/conv3_3/Conv2D:0')
        feat4 = sess.run(output4, feed_dict=feed_dict)
        #output5 = graph.get_tensor_by_name('vgg_16/conv4/conv4_2/Conv2D:0')
        #feat5 = sess.run(output5, feed_dict=feed_dict)
        output6 = graph.get_tensor_by_name('vgg_16/conv4/conv4_3/Conv2D:0')
        feat6 = sess.run(output6, feed_dict=feed_dict)
        #output7 = graph.get_tensor_by_name('vgg_16/conv5/conv5_2/Conv2D:0')
        #feat7 = sess.run(output7, feed_dict=feed_dict)
        output8 = graph.get_tensor_by_name('vgg_16/conv5/conv5_3/Conv2D:0')
        feat8 = sess.run(output8, feed_dict=feed_dict)


        #feat88 = tf.concat([feat88, feat8.view().reshape(feat8.shape[0], -1)],0)
        #feat88 = np.concatenate([feat88, feat8.view().reshape(feat8.shape[0], -1)],0)
        #output9 = graph.get_tensor_by_name('resnet_v1_101_2/block3/unit_23/bottleneck_v1/conv3/Conv2D:0')
        #feat9 = sess.run(output9, feed_dict=feed_dict)
        #output9 = graph.get_tensor_by_name('resnet_v1_101_3/rpn_conv/3x3/Conv2D:0')
        #feat9 = sess.run(output9, feed_dict=feed_dict)
        #feat99 = tf.concat([feat99, feat9.view().reshape(feat9.shape[0], -1)],0)
        #feat99 = np.concatenate([feat99, feat9.view().reshape(feat9.shape[0], -1)],0)
        with tf.device("/cpu:0"):
            feat00 = np.concatenate([feat00, feat0.view().reshape(feat0.shape[0], -1)],0)
            feat11 = np.concatenate([feat11, feat1.view().reshape(feat1.shape[0], -1)],0)
            feat22 = np.concatenate([feat22, feat2.view().reshape(feat2.shape[0], -1)],0)
            feat33 = np.concatenate([feat33, feat3.view().reshape(feat3.shape[0], -1)],0)
            feat44 = np.concatenate([feat44, feat4.view().reshape(feat4.shape[0], -1)],0)
            feat55 = np.concatenate([feat55, feat5.view().reshape(feat5.shape[0], -1)],0)
            feat66 = np.concatenate([feat66, feat6.view().reshape(feat6.shape[0], -1)],0)
            feat77 = np.concatenate([feat77, feat7.view().reshape(feat7.shape[0], -1)],0)
            feat88 = np.concatenate([feat88, feat8.view().reshape(feat8.shape[0], -1)],0)
            #feat99 = np.concatenate([feat99, feat9.view().reshape(feat9.shape[0], -1)],0)"""

        """ID = {
                  "feat00": feat00,
                  "feat11": feat11,
                  "feat22": feat22,
                  "feat33": feat33,
                  "feat44": feat44,
                  "feat55": feat55,
                  "feat66": feat66,
                  "feat77": feat77,
                  "feat88": feat88,
                  }"""
    #print(feat1.shape)
    #print(feat2.shape)
    #print(feat3.shape)
    #print(feat4.shape)
    #print(feat5.shape)
    #print(feat6.shape)
    #print(feat7.shape)
    #print(feat8.shape)
    #print(feat9.shape)
    #print(feat10.shape)
    #print(feat11_.shape)

    def vis_detections(im, class_name, dets, noise, thresh=0.5):
        #Draw detected bounding boxes.
        #print(dets)
        #print(dets.shape)
        #print(dets[:, -1])
        inds = np.where(dets[:, -1] >= 0.8)[0]
        #print(thresh)
        print(inds)
        if len(inds) == 0:
            return
        im = im[:, :, (2, 1, 0)]
        fig, ax = plt.subplots(figsize=(12, 12))
        #print(im.shape)
        #fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.imshow(im, aspect='equal')
        for ind in inds:
            #print("now here")
            bbox = dets[ind, :4]
            score = dets[ind, -1]

            ax.add_patch(
                    plt.Rectangle((bbox[0], bbox[1]),
                        bbox[2] - bbox[0],
                        bbox[3] - bbox[1], fill=False,
                        edgecolor='red', linewidth=3.0)
                    )
            ax.text(bbox[0], bbox[1] - 2,
                    '{:.3f}'.format(score),
                    bbox=dict(facecolor='blue', alpha=0.5),
                    fontsize=15, color='white')
        #ax.set_title(('{} detections with '
        #    'p({} | box) >= {:.1f}').format(class_name, class_name,thresh),
        #          fontsize=14)
        plt.axis('off')
        #ax.margins(x=0)
        #plt.margins(x=0)
        plt.rcParams['axes.xmargin'] = 0
        plt.rcParams['axes.ymargin'] = 0
        plt.tight_layout()
        print(i)
        #cv2.imwrite("{}.jpg".format(i), wavelet)
        #plt.savefig(str(i) +'.jpg')
        plt.savefig(str(i) + noise +'.jpg', bbox_inches='tight', pad_inches=0)
        #plt.imsave(str(i)+'.jpg', )
        #plt.savefig(figsize((800/my_dpi, 800/my_dpi), dpi=my_dpi))
        #plt.savefig(str(i) +'.jpg', dpi=my_dpi)

    """# Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(imdb.classes[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, noise, thresh=CONF_THRESH)"""
  #with tf.device("/gpu:0"):
  """ID_all = []
  count = 1
  for key, value in sorted(ID.iteritems()):
      #print(key, value)
      ID_cal = net.computeID(value, 20, 0.9)
      ID_all.append(ID_cal)
      print('completed {}: {}'.format(count, key))
      wandb.log({"ID": ID_cal[0], "realtive layer depth": count})
      count+=1
  ID_all = np.array(ID_all)
  print("Final result: {}".format(ID_all[:,0]))
  #ID_final = ID_all[:,0]
  #wandb.log({"ID": ID_final})
  print("Done.")"""


  det_file = os.path.join(output_dir, 'detections.pkl')
  with open(det_file, 'wb') as f:
      pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

  print('Evaluating detections')
  imdb.evaluate_detections(all_boxes, output_dir)
