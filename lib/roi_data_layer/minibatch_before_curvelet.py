# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
import cv2
from model.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob
from skimage.util import random_noise
from skimage.restoration import (denoise_wavelet, estimate_sigma)
from sklearn.cluster import MiniBatchKMeans
from scipy.stats import uniform
from scipy.stats import gamma
from scipy.stats import rayleigh
from skimage import img_as_float
import random
import wandb
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from numpy import zeros, newaxis
#from skimage.metrics import peak_signal_noise_ratio

from PIL import Image
from PIL import ImageEnhance

def get_minibatch(roidb, num_classes):
  """Given a roidb, construct a minibatch sampled from it."""
  num_images = len(roidb)
  # Sample random scales to use for each image in this batch
  random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                  size=num_images)
  assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
    'num_images ({}) must divide BATCH_SIZE ({})'. \
    format(num_images, cfg.TRAIN.BATCH_SIZE)

  # Get the input image blob, formatted for caffe
  im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)

  blobs = {'data': im_blob}

  assert len(im_scales) == 1, "Single batch only"
  assert len(roidb) == 1, "Single batch only"
  
  # gt boxes: (x1, y1, x2, y2, cls)
  if cfg.TRAIN.USE_ALL_GT:
    # Include all ground truth boxes
    gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
  else:
    # For the COCO ground truth boxes, exclude the ones that are ''iscrowd'' 
    gt_inds = np.where(roidb[0]['gt_classes'] != 0 & np.all(roidb[0]['gt_overlaps'].toarray() > -1.0, axis=1))[0]
  gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
  gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
  gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
  blobs['gt_boxes'] = gt_boxes
  blobs['im_info'] = np.array(
    [im_blob.shape[1], im_blob.shape[2], im_scales[0]],
    dtype=np.float32)

  return blobs

def _get_image_blob(roidb, scale_inds):
  """Builds an input blob from the images in the roidb at the specified
  scales.
  """
  num_images = len(roidb)
  processed_ims = []
  im_scales = []
  for i in range(num_images):
    img = cv2.imread(roidb[i]['image'])
    
    def add_gaussian_noise(noise_type):
        if ('gaussian_wavelet' in noise_type):
            if ('var0.4' in noise_type):
                im_noise = random_noise(img, mode='gaussian', var=0.4)
                im_bayes = denoise_wavelet(im_noise, method='BayesShrink', mode='soft',
                                    wavelet_levels=3,
                                    multichannel= True, convert2ycbcr=True)
                data = (255 * im_bayes)
                im = data.astype(np.uint8)
                print('gaussian wavelet var 0.4')
            elif ('var1.0' in noise_type):
                im_noise = random_noise(img, mode='gaussian', var=1.0)
                im_bayes = denoise_wavelet(im_noise, method='BayesShrink', mode='soft',
                                    wavelet_levels=3,
                                    multichannel= True, convert2ycbcr=True)
                data = (255 * im_bayes)
                im = data.astype(np.uint8)
                print('gaussian wavelet var 1.0')
            elif ('var1.5' in noise_type):
                im_noise = random_noise(img, mode='gaussian', var=1.5)
                im_bayes = denoise_wavelet(im_noise, method='BayesShrink', mode='soft',
                                    wavelet_levels=3,
                                    multichannel= True, convert2ycbcr=True)
                data = (255 * im_bayes)
                im = data.astype(np.uint8)
                print('gaussian wavelet var 1.5')
        elif('gaussian_gausblur' in noise_type):
            size = 3
            if ('var0.4' in noise_type):
                gauss_array = random_noise(img, mode='gaussian', var=0.4)
                im_noise = (255 * gauss_array).astype(np.uint8)
                im = cv2.GaussianBlur(im_noise, (size, size), 0)
                print('gaussian blur var 0.4')
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
            if ('var0.4' in noise_type):
                gauss_array = random_noise(img, mode='gaussian', var=0.4)
                im_noise = (255 * gauss_array).astype(np.uint8)
                im = cv2.blur(im_noise, (size, size))
                print('gaussian mean var 0.4')
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
            if ('var0.4' in noise_type):
                gauss_array = random_noise(img, mode='gaussian', var=0.4)
                im_noise = (255 * gauss_array).astype(np.uint8)
                im = cv2.medianBlur(im_noise, size)
                print('gaussian median var 0.4')
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
            if ('var0.4' in noise_type):
                gauss_array = random_noise(img, mode='gaussian', var=0.4)
                im_noise = (255 * gauss_array).astype(np.uint8)
                im = cv2.bilateralFilter(im_noise, diameter, sigmaColor, sigmaSpace, borderType=cv2.BORDER_CONSTANT)
                print('gaussian bilateral var 0.4')
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
            if ('var0.4' in noise_type):
                gauss_array = random_noise(img, mode='gaussian', var=0.4)
                im = (255 * gauss_array).astype(np.uint8)
                print('gaussian var 0.4')
            elif ('var1.0' in noise_type):
                gauss_array = random_noise(img, mode='gaussian', var=1.0)
                im = (255 * gauss_array).astype(np.uint8)
                print('gaussian var 1.0')
            elif ('var1.5' in noise_type):
                gauss_array = random_noise(img, mode='gaussian', var=1.5)
                im = (255 * gauss_array).astype(np.uint8)
                print('gaussian var 1.5')
        print("Gaussian")
        return im 

    def add_poisson_noise(noise_type):
        if ('poisson' in noise_type):
        #introduce poisson noise.
        #also called shot noise originates from the discrete nature of electronic charge or photons.
            if ('poisson_wavelet' in noise_type):
                pois_array = random_noise(img, mode='poisson')
                im_noise = (255 * pois_array).astype(np.uint8)
                im_bayes = denoise_wavelet(im_noise, method='BayesShrink', mode='soft',
                                        wavelet_levels=3,
                                        multichannel= True, convert2ycbcr=True)
                data = (255 * im_bayes)
                im = data.astype(np.uint8)
                print('poisson wavelet')
            elif('poisson_gausblur' in noise_type):
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
                im = (255 * pois_array).astype(np.uint8)
                print('poisson noise')
        return im

    def add_sap_noise(noise_type):
        if ('sap' in noise_type):
            img = cv2.imread(roidb[i]['image'])
            if ('sap_wavelet' in noise_type):
                if ('var0.2' in noise_type):
                    sp_array = random_noise(img, mode='s&p', amount=0.2)
                    im_bayes = denoise_wavelet(sp_array, method='BayesShrink', mode='soft',
                                        wavelet_levels=3,
                                        multichannel= True, convert2ycbcr=True)
                    data = (255 * im_bayes)
                    im = data.astype(np.uint8)
                    print('wavelet')
                    print('s&p wavelet var 0.2')
                elif ('var0.4' in noise_type):
                    sp_array = random_noise(img, mode='s&p', amount=0.4)
                    im_bayes = denoise_wavelet(sp_array, method='BayesShrink', mode='soft',
                                        wavelet_levels=3,
                                        multichannel= True, convert2ycbcr=True)
                    data = (255 * im_bayes)
                    im = data.astype(np.uint8)
                    print('wavelet')
                    print('s&p wavelet var 0.4')
                elif ('var0.8' in noise_type):
                    sp_array = random_noise(img, mode='s&p', amount=0.8)
                    im_bayes = denoise_wavelet(sp_array, method='BayesShrink', mode='soft',
                                        wavelet_levels=3,
                                        multichannel= True, convert2ycbcr=True)
                    data = (255 * im_bayes)
                    im = data.astype(np.uint8)
                    print('wavelet')
                    print('s&p wavelet var 0.8')
            elif('sap_gausblur' in noise_type):
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
                    im = (255 * sp_array).astype(np.uint8)
                    print('s&p var 0.2')
                elif ('var0.4' in noise_type):
                    sp_array = random_noise(img, mode='s&p', amount=0.4)
                    im = (255 * sp_array).astype(np.uint8)
                    print('s&p var 0.4')
                elif ('var0.8' in noise_type):
                    sp_array = random_noise(img, mode='s&p', amount=0.8)
                    im = (255 * sp_array).astype(np.uint8)
                    print('s&p var 0.8')
            print("salt & pepper")
        return im

    def add_speckle_noise(noise_type):
        if ('speckle' in noise_type):
            img = cv2.imread(roidb[i]['image'])
            if ('speckle_wavelet' in noise_type):
                if ('var0.5' in noise_type):
                    speck_array = random_noise(img, mode='speckle', var=0.5)
                    im_bayes = denoise_wavelet(speck_array, method='BayesShrink', mode='soft',
                                        wavelet_levels=3,
                                        multichannel= True, convert2ycbcr=True)
                    data = (255 * im_bayes)
                    im = data.astype(np.uint8)
                    print('speckle wavelet var 0.4')
                elif ('var1.0' in noise_type):
                    speck_array = random_noise(img, mode='speckle', var=1.0)
                    im_bayes = denoise_wavelet(speck_array, method='BayesShrink', mode='soft',
                                        wavelet_levels=3,
                                        multichannel= True, convert2ycbcr=True)
                    data = (255 * im_bayes)
                    im = data.astype(np.uint8)
                    print('speckle wavelet var 1.0')
                elif ('var2.0' in noise_type):
                    speck_array = random_noise(img, mode='speckle', var=2.0)
                    im_bayes = denoise_wavelet(speck_array, method='BayesShrink', mode='soft',
                                        wavelet_levels=3,
                                        multichannel= True, convert2ycbcr=True)
                    data = (255 * im_bayes)
                    im = data.astype(np.uint8)
                    print('speckle wavelet var 2.0')
            elif('speckle_gausblur' in noise_type):
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
                    im = (255 * speck_array).astype(np.uint8)
                    print('speckle var 0.5')
                elif ('var1.0' in noise_type):
                    speck_array = random_noise(img, mode='speckle', var=1.0)
                    im = (255 * speck_array).astype(np.uint8)
                    print('speckle var 1.0')
                elif ('var2.0' in noise_type):
                    speck_array = random_noise(img, mode='speckle', var=2.0)
                    im = (255 * speck_array).astype(np.uint8)
                    print('speckle var 2.0')
            print("Speckle")
        return im

    def add_quant_noise(noise_type):
        if ('quant' in noise_type):
            img = cv2.imread(roidb[i]['image'])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            h, w = img.shape[:2]
            #clor quantization, using K-Means clustering.
            #Usually this noise is found while converting analog to digital, or continuous random variable to discreate.
            image = img.reshape((img.shape[0] * img.shape[1], 3))
            if ('quant_wavelet' in noise_type):
                if ('var3' in noise_type):
                    clt = MiniBatchKMeans(n_clusters= 3)
                    labels = clt.fit_predict(image)
                    quant = clt.cluster_centers_.astype("uint8")[labels]
                    quant = quant.reshape(h, w, 3)
                    quant_array = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
                    im_bayes = denoise_wavelet(quant_array, method='BayesShrink', mode='soft',
                                        wavelet_levels=3,
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
                                        wavelet_levels=3,
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
                                        wavelet_levels=3,
                                        multichannel= True, convert2ycbcr=True)
                    data = (255 * im_bayes)
                    im = data.astype(np.uint8)
                    print('quantization wavelet with cluster 10')
            elif('quant_gausblur' in noise_type):
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
            img = cv2.imread(roidb[i]['image'])
            image = img_as_float(img)
            if ('uniform_wavelet' in noise_type):
                if ('var0.2' in noise_type):
                    uniform_array = np.random.uniform(low=0., high=0.2, size=img.shape)
                    im_noise = cv2.add(image, uniform_array)
                    im_bayes = denoise_wavelet(im_noise, method='BayesShrink', mode='soft',
                                        wavelet_levels=3,
                                        multichannel= True, convert2ycbcr=True)
                    data = (255 * im_bayes)
                    im = data.astype(np.uint8)
                    print('wavelet')
                    print('uniform wavelet var 0.2')
                elif ('var0.6' in noise_type):
                    uniform_array = np.random.uniform(low=0., high=0.6, size=img.shape)
                    im_noise = cv2.add(image, uniform_array)
                    im_bayes = denoise_wavelet(im_noise, method='BayesShrink', mode='soft',
                                        wavelet_levels=3,
                                        multichannel= True, convert2ycbcr=True)
                    data = (255 * im_bayes)
                    im = data.astype(np.uint8)
                    print('wavelet')
                    print('uniform wavelet var 0.6')
                elif ('var1.2' in noise_type):
                    uniform_array = np.random.uniform(low=0., high=1.2, size=img.shape)
                    im_noise = cv2.add(image, uniform_array)
                    im_bayes = denoise_wavelet(im_noise, method='BayesShrink', mode='soft',
                                        wavelet_levels=3,
                                        multichannel= True, convert2ycbcr=True)
                    data = (255 * im_bayes)
                    im = data.astype(np.uint8)
                    print('wavelet')
                    print('uniform wavelet var 1.2')
            elif('uniform_gausblur' in noise_type):
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
                    im = (255 * uniform_noise).astype(np.uint8)
                    print('uniform var 0.2')
                elif ('var0.6' in noise_type):
                    uniform_array = np.random.uniform(low=0., high=0.6, size=img.shape)
                    uniform_noise = cv2.add(image, uniform_array)
                    im = (255 * uniform_noise).astype(np.uint8)
                    print('uniform var 0.6')
                elif ('var1.2' in noise_type):
                    uniform_array = np.random.uniform(low=0., high=1.2, size=img.shape)
                    uniform_noise = cv2.add(image, uniform_array)
                    im = (255 * uniform_noise).astype(np.uint8)
                    print('uniform var 1.2')
            print("uniform")
        return im

    def add_brownian_noise(noise_type):
        if ('brownian' in noise_type):
            img = cv2.imread(roidb[i]['image'])
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
                                        wavelet_levels=3,
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
                                        wavelet_levels=3,
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
                                        wavelet_levels=3,
                                        multichannel= True, convert2ycbcr=True)
                    data = (255 * im_bayes)
                    im = data.astype(np.uint8)
                    print('brownian wavelet var 0.009')
            elif('brownian_gausblur' in noise_type):
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
            img = cv2.imread(roidb[i]['image'])
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
                                        wavelet_levels=3,
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
                                        wavelet_levels=3,
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
                                        wavelet_levels=3,
                                        multichannel= True, convert2ycbcr=True)
                    data = (255 * im_bayes)
                    im = data.astype(np.uint8)
                    print('periodic wavelet amplitude size')
            elif('periodic_gausblur' in noise_type):
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
            img = cv2.imread(roidb[i]['image'])
            image = img_as_float(img)
            a = 1.99
            if ('gamma_wavelet' in noise_type):
                if ('var0.1' in noise_type):
                    gamma_dist = gamma.rvs(a, loc=0., scale=0.1, size=image.shape)
                    gamma_array = cv2.add(image, gamma_dist)
                    im_bayes = denoise_wavelet(gamma_array, method='BayesShrink', mode='soft',
                                        wavelet_levels=3,
                                        multichannel= True, convert2ycbcr=True)
                    data = (255 * im_bayes)
                    im = data.astype(np.uint8)
                    print('gamma wavelet var 0.1')
                elif ('var0.3' in noise_type):
                    gamma_dist = gamma.rvs(a, loc=0., scale=0.3, size=image.shape)
                    gamma_array = cv2.add(image, gamma_dist)
                    im_bayes = denoise_wavelet(gamma_array, method='BayesShrink', mode='soft',
                                        wavelet_levels=3,
                                        multichannel= True, convert2ycbcr=True)
                    data = (255 * im_bayes)
                    im = data.astype(np.uint8)
                    print('gamma wavelet var 0.3')
                elif ('var0.7' in noise_type):
                    gamma_dist = gamma.rvs(a, loc=0., scale=0.7, size=image.shape)
                    gamma_array = cv2.add(image, gamma_dist)
                    im_bayes = denoise_wavelet(gamma_array, method='BayesShrink', mode='soft',
                                        wavelet_levels=3,
                                        multichannel= True, convert2ycbcr=True)
                    data = (255 * im_bayes)
                    im = data.astype(np.uint8)
                    print('gamma wavelet var 0.7')
            elif('gamma_gausblur' in noise_type):
                size = 3
                if ('var0.1' in noise_type):
                    gamma_dist = gamma.rvs(a, loc=0., scale=0.1, size=image.shape)
                    gamma_array = cv2.add(image, gamma_dist)
                    im_noise = (gamma_array * 255).astype(np.uint8)
                    im = cv2.GaussianBlur(im_noise, (size, size), 0)
                    print('gamma gausblur var 0.1')
                elif ('var0.3' in noise_type):
                    gamma_dist = gamma.rvs(a, loc=0., scale=0.3, size=image.shape)
                    gamma_array = cv2.add(image, gamma_dist)
                    im_noise = (gamma_array * 255).astype(np.uint8)
                    im = cv2.GaussianBlur(im_noise, (size, size), 0)
                    print('gamma gausblur var 0.3')
                elif ('var0.7' in noise_type):
                    gamma_dist = gamma.rvs(a, loc=0., scale=0.7, size=image.shape)
                    gamma_array = cv2.add(image, gamma_dist)
                    im_noise = (gamma_array * 255).astype(np.uint8)
                    im = cv2.GaussianBlur(im_noise, (size, size), 0)
                    print('gamma gausblur var 0.7')
            elif('gamma_mean' in noise_type):
                size = 3
                if ('var0.1' in noise_type):
                    gamma_dist = gamma.rvs(a, loc=0., scale=0.1, size=image.shape)
                    gamma_array = cv2.add(image, gamma_dist)
                    im_noise = (gamma_array * 255).astype(np.uint8)
                    im = cv2.blur(im_noise, (size, size))
                    print('gamma mean var 0.1')
                elif ('var0.3' in noise_type):
                    gamma_dist = gamma.rvs(a, loc=0., scale=0.3, size=image.shape)
                    gamma_array = cv2.add(image, gamma_dist)
                    im_noise = (gamma_array * 255).astype(np.uint8)
                    im = cv2.blur(im_noise, (size, size))
                    print('gamma mean var 0.3')
                elif ('var0.7' in noise_type):
                    gamma_dist = gamma.rvs(a, loc=0., scale=0.7, size=image.shape)
                    gamma_array = cv2.add(image, gamma_dist)
                    im_noise = (gamma_array * 255).astype(np.uint8)
                    im = cv2.blur(im_noise, (size, size))
                    print('gamma mean var 0.7')
            elif('gamma_median' in noise_type):
                size = 3
                if ('var0.1' in noise_type):
                    gamma_dist = gamma.rvs(a, loc=0., scale=0.1, size=image.shape)
                    gamma_array = cv2.add(image, gamma_dist)
                    im_noise = (gamma_array * 255).astype(np.uint8)
                    im = cv2.medianBlur(im_noise, size)
                    print('gamma median var 0.1')
                elif ('var0.3' in noise_type):
                    gamma_dist = gamma.rvs(a, loc=0., scale=0.3, size=image.shape)
                    gamma_array = cv2.add(image, gamma_dist)
                    im_noise = (gamma_array * 255).astype(np.uint8)
                    im = cv2.medianBlur(im_noise, size)
                    print('gamma median var 0.3')
                elif ('var0.7' in noise_type):
                    gamma_dist = gamma.rvs(a, loc=0., scale=0.7, size=image.shape)
                    gamma_array = cv2.add(image, gamma_dist)
                    im_noise = (gamma_array * 255).astype(np.uint8)
                    im = cv2.medianBlur(im_noise, size)
                    print('gamma median var 0.7')
            elif('gamma_bilateral' in noise_type):
                diameter = 9      #the diameter of each pixel in the neighborhood used during filtering
                sigmaColor = 20     #sigma of grey/color space.
                sigmaSpace = 100    #Large value means farther pixels influence each other.
                if ('var0.1' in noise_type):
                    gamma_dist = gamma.rvs(a, loc=0., scale=0.1, size=image.shape)
                    gamma_array = cv2.add(image, gamma_dist)
                    im_noise = (gamma_array * 255).astype(np.uint8)
                    im = cv2.bilateralFilter(im_noise, diameter, sigmaColor, sigmaSpace, borderType=cv2.BORDER_CONSTANT)
                    print('gamma bilateral var 0.1')
                elif ('var0.3' in noise_type):
                    gamma_dist = gamma.rvs(a, loc=0., scale=0.3, size=image.shape)
                    gamma_array = cv2.add(image, gamma_dist)
                    im_noise = (gamma_array * 255).astype(np.uint8)
                    im = cv2.bilateralFilter(im_noise, diameter, sigmaColor, sigmaSpace, borderType=cv2.BORDER_CONSTANT)
                    print('gamma bilateral var 0.3')
                elif ('var0.7' in noise_type):
                    gamma_dist = gamma.rvs(a, loc=0., scale=0.7, size=image.shape)
                    gamma_array = cv2.add(image, gamma_dist)
                    im_noise = (gamma_array * 255).astype(np.uint8)
                    im = cv2.bilateralFilter(im_noise, diameter, sigmaColor, sigmaSpace, borderType=cv2.BORDER_CONSTANT)
                    print('gamma bilateral var 0.7')
            else:
                if ('var0.1' in noise_type):
                    gamma_dist = gamma.rvs(a, loc=0., scale=0.1, size=image.shape)
                    gamma_array = cv2.add(image, gamma_dist)
                    im = (gamma_array * 255).astype(np.uint8)
                    print('gamma var 0.1')
                elif ('var0.3' in noise_type):
                    gamma_dist = gamma.rvs(a, loc=0., scale=0.3, size=image.shape)
                    gamma_array = cv2.add(image, gamma_dist)
                    im = (gamma_array * 255).astype(np.uint8)
                    print('gamma var 0.3')
                elif ('var0.7' in noise_type):
                    gamma_dist = gamma.rvs(a, loc=0., scale=0.7, size=image.shape)
                    gamma_array = cv2.add(image, gamma_dist)
                    im = (gamma_array * 255).astype(np.uint8)
                    print('gamma var 0.7')
            print("Gamma")
        return im

    def add_rayleigh_noise(noise_type):
        if ('rayleigh' in noise_type):
            img = cv2.imread(roidb[i]['image'])
            image = img_as_float(img)
            if ('rayleigh_wavelet' in noise_type):
                if ('var0.1' in noise_type):
                    rayleigh_dist = rayleigh.rvs(loc=0., scale=0.1, size=image.shape)
                    rayleigh_array = cv2.add(image, rayleigh_dist)
                    im_bayes = denoise_wavelet(rayleigh_array, method='BayesShrink', mode='soft',
                                        wavelet_levels=3,
                                        multichannel= True, convert2ycbcr=True)
                    data = (255 * im_bayes)
                    im = data.astype(np.uint8)
                    print('rayleigh wavelet var 0.1')
                elif ('var0.2' in noise_type):
                    rayleigh_dist = rayleigh.rvs(loc=0., scale=0.2, size=image.shape)
                    rayleigh_array = cv2.add(image, rayleigh_dist)
                    im_bayes = denoise_wavelet(rayleigh_array, method='BayesShrink', mode='soft',
                                        wavelet_levels=3,
                                        multichannel= True, convert2ycbcr=True)
                    data = (255 * im_bayes)
                    im = data.astype(np.uint8)
                    print('rayleigh wavelet var 0.2')
                elif ('var0.3' in noise_type):
                    rayleigh_dist = rayleigh.rvs(loc=0., scale=0.3, size=image.shape)
                    rayleigh_array = cv2.add(image, rayleigh_dist)
                    im_bayes = denoise_wavelet(rayleigh_array, method='BayesShrink', mode='soft',
                                        wavelet_levels=3,
                                        multichannel= True, convert2ycbcr=True)
                    data = (255 * im_bayes)
                    im = data.astype(np.uint8)
                    print('rayleigh wavelet var 0.3')
            elif('rayleigh_gausblur' in noise_type):
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
                    im = (rayleigh_array * 255).astype(np.uint8)
                    print('rayleigh var 0.1')
                elif ('var0.2' in noise_type):
                    rayleigh_dist = rayleigh.rvs(loc=0., scale=0.2, size=image.shape)
                    rayleigh_array = cv2.add(image, rayleigh_dist)
                    im = (rayleigh_array * 255).astype(np.uint8)
                    print('rayleigh var 0.2')
                elif ('var0.3' in noise_type):
                    rayleigh_dist = rayleigh.rvs(loc=0., scale=0.3, size=image.shape)
                    rayleigh_array = cv2.add(image, rayleigh_dist)
                    im = (rayleigh_array * 255).astype(np.uint8)
                    print('rayleigh var 0.3')
            print("Rayleigh")
        return im
    def add_bloom():
        img = cv2.imread(roidb[i]['image'])
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
        print('bloom')
        return im
    def add_shader():
        factor = 3
        im = Image.open(roidb[i]['image'])
        im_out = ImageEnhance.Brightness(im).enhance(factor)
        im = np.array(im_out)
        print('PIL enhance')
        return im
    def retain_original():
        im = cv2.imread(roidb[i]['image'])
        #introduce gaussian noise.
        print('original')
        return im

    if ('mix' in roidb[i]['noise_type']):
        if ('var_low' in roidb[i]['noise_type']):
            noise_list = ['gaussian_var0.4', 'poisson', 'speckle_var0.5',
                            'sap_var0.2', 'uniform_var0.2', 'gamma_var0.1',
                            'rayleigh_var0.1','periodic_var3.14','brownian_var0.9', 'quant_var3']
            noise_type = random.choice(noise_list)
            print(noise_type)
        elif ('var_medium' in roidb[i]['noise_type']):
            noise_list = ['gaussian_var1.0', 'poisson', 'speckle_var1.0',
                            'sap_var0.4', 'uniform_var0.6', 'gamma_var0.3',
                            'rayleigh_var0.2','periodic_var100','brownian_var0.09', 'quant_var7', 'original', 'shader']
            noise_type = random.choice(noise_list)
        elif ('var_high' in roidb[i]['noise_type']):
            noise_list = ['gaussian_var1.5', 'poisson', 'speckle_var2.0',
                            'sap_var0.8', 'uniform_var1.2', 'gamma_var0.7',
                            'rayleigh_var0.3','periodic_varsize','brownian_var0.009', 'quant_var10', 'original']
            noise_type = random.choice(noise_list)
        elif ('var_all' in roidb[i]['noise_type']):
            noise_list = ['gaussian_var0.4', 'poisson', 'speckle_var0.5',
                            'sap_var0.2', 'uniform_var0.2', 'gamma_var0.1',
                            'rayleigh_var0.1','periodic_var3.14','brownian_var0.9', 'quant_var3', 'original', 
                            'gaussian_var1.0', 'poisson', 'speckle_var1.0',
                            'sap_var0.4', 'uniform_var0.6', 'gamma_var0.3',
                            'rayleigh_var0.2','periodic_var100','brownian_var0.09', 'quant_var7'
                            'gaussian_var1.5', 'poisson', 'speckle_var2.0',
                            'sap_var0.8', 'uniform_var1.2', 'gamma_var0.7',
                            'rayleigh_var0.3','periodic_varsize','brownian_var0.009', 'quant_var10', 'original']
            noise_type = random.choice(noise_list)

        if ('gaussian' in noise_type):
            im = add_gaussian_noise(noise_type)
        elif('poisson' in noise_type):
            im = add_poisson_noise(noise_type)
        elif('sap' in noise_type):
            im = add_sap_noise(noise_type)
        elif('speckle' in noise_type):
            im = add_speckle_noise(noise_type)
        elif('periodic' in noise_type):
            im = add_periodic_noise(noise_type)
        elif('brownian' in noise_type):
            im = add_brownian_noise(noise_type)
        elif('quant' in noise_type):
            im = add_quant_noise(noise_type)
        elif('uniform' in noise_type):
            im = add_uniform_noise(noise_type)
        elif('gamma' in noise_type):
            im = add_gamma_noise(noise_type)
        elif('rayleigh' in noise_type):
            im = add_rayleigh_noise(noise_type)
        elif('shader' in noise_type):
            im = add_shader()
        else :
            im = retain_original()
    elif ('gaussian' in roidb[i]['noise_type']):
        noise_type = roidb[i]['noise_type']
        im = add_gaussian_noise(noise_type)
    elif('poisson' in roidb[i]['noise_type']):
        noise_type = roidb[i]['noise_type']
        im = add_poisson_noise(noise_type)
    elif('sap' in roidb[i]['noise_type']):
        noise_type = roidb[i]['noise_type']
        im = add_sap_noise(noise_type)
    elif('speckle' in roidb[i]['noise_type']):
        noise_type = roidb[i]['noise_type']
        im = add_speckle_noise(noise_type)
    elif('periodic' in roidb[i]['noise_type']):
        noise_type = roidb[i]['noise_type']
        im = add_periodic_noise(noise_type)
    elif('brownian' in roidb[i]['noise_type']):
        noise_type = roidb[i]['noise_type']
        im = add_brownian_noise(noise_type)
    elif('quant' in roidb[i]['noise_type']):
        noise_type = roidb[i]['noise_type']
        im = add_quant_noise(noise_type)
    elif('uniform' in roidb[i]['noise_type']):
        noise_type = roidb[i]['noise_type']
        im = add_uniform_noise(noise_type)
    elif('gamma' in roidb[i]['noise_type']):
        noise_type = roidb[i]['noise_type']
        im = add_gamma_noise(noise_type)
    elif('rayleigh' in roidb[i]['noise_type']):
        noise_type = roidb[i]['noise_type']
        im = add_rayleigh_noise(noise_type)
    elif('bloom' in roidb[i]['noise_type']):
        noise_type = roidb[i]['noise_type']
        im = add_bloom()
    elif('shader' in roidb[i]['noise_type']):
        noise_type = roidb[i]['noise_type']
        im = add_shader()
    else :
        im = retain_original()

    if ('gaus_blur' in roidb[i]['noise_type']):
        k_size = 3
        im = cv2.GaussianBlur(im, (k_size, k_size), 0)
        print('gaussian blur mix')
    elif ('mean' in roidb[i]['noise_type']):
        k_size = 3
        im = cv2.blur(im, (k_size, k_size))
        print('mean mix')
    elif ('median' in roidb[i]['noise_type']):
        k_size = 3
        im_noise = (im * 255).astype(np.uint8)
        im = cv2.medianBlur(im_noise, k_size)
        print('median mix')
    elif ('wavelet' in roidb[i]['noise_type']):
        """im_bayes = denoise_wavelet(im, method='BayesShrink', mode='soft',
                                    wavelet_levels=3,
                                    multichannel= True, convert2ycbcr=True)"""
        im_bayes = denoise_wavelet(im, method='BayesShrink', mode='soft',
                                    multichannel= True, convert2ycbcr=True)
        data = (255 * im_bayes)
        im = data.astype(np.uint8)
        print('wavelet mix')
    elif ('bilateral' in roidb[i]['noise_type']):
        diameter = 9      #the diameter of each pixel in the neighborhood used during filtering
        sigmaColor = 20     #sigma of grey/color space.
        sigmaSpace = 100    #Large value means farther pixels influence each other.
        im = cv2.bilateralFilter(im, diameter, sigmaColor, sigmaSpace, borderType=cv2.BORDER_CONSTANT)
        print('bilateral mix')
  
  
    if roidb[i]['flipped']:
      im = im[:, ::-1, :]
    target_size = cfg.TRAIN.SCALES[scale_inds[i]]
    im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                    cfg.TRAIN.MAX_SIZE)
    im_scales.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, im_scales
