import curvelops as cl
import cv2
from skimage import img_as_float
import numpy as np
from skimage.util import random_noise
import sys

class FDCT:
    def __init__(self, noise, img_path):
        self.noise = noise
        #self.img = cv2.imread("/home/mahesh/thesis/de-noise/tf-faster-rcnn/data/6thfloorData/6thFloorTest/JPEGImages/I004401.jpg")
        self.img = cv2.imread(img_path)

    def add_gaussian_noise(self, noise_type):
        im_noise = np.array([])
        self.img = cv2.imread("/home/mahesh/thesis/de-noise/tf-faster-rcnn/data/6thfloorData/6thFloorTest/JPEGImages/I004401.jpg")
        if ('var0.4' in noise_type):
            im_noise = random_noise(self.img, mode='gaussian', var=0.4)
        elif ('var1.0' in noise_type):
            im_noise = random_noise(self.img, mode='gaussian', var=1.0)
        elif ('var1.5' in noise_type):
            im_noise = random_noise(self.img, mode='gaussian', var=1.5)
        return im_noise

    def add_poisson_noise(self, noise_type):
        im_noise = np.array([])
        if ('poisson' in noise_type):
            im_noise = random_noise(self.img, mode='poisson')
            print('poisson noise')
        return im_noise

    def add_sap_noise(self, noise_type):
        im_noise = np.array([])
        if ('var0.2' in noise_type):
            im_noise = random_noise(self.img, mode='s&p', amount=0.2)
            print('s&p wavelet var 0.2')
        elif ('var0.4' in noise_type):
            im_noise = random_noise(self.img, mode='s&p', amount=0.4)
            print('s&p wavelet var 0.4')
        elif ('var0.8' in noise_type):
            im_noise = random_noise(self.img, mode='s&p', amount=0.8)
            print('s&p wavelet var 0.8')
        return im_noise

    def add_speckle_noise(self, noise_type):
        im_noise = np.array([])
        if ('var0.5' in noise_type):
            im_noise = random_noise(self.img, mode='speckle', var=0.5)
            print('speckle var 0.5')
        elif ('var1.0' in noise_type):
            im_noise = random_noise(self.img, mode='speckle', var=1.0)
            print('speckle var 1.0')
        elif ('var2.0' in noise_type):
            im_noise = random_noise(self.img, mode='speckle', var=2.0)
            print('speckle var 2.0')
        return im_noise

    def add_uniform_noise(self, noise_type):
        im_noise = np.array([])
        self.image = img_as_float(self.img)
        if ('var0.2' in noise_type):
            uniform_array = np.random.uniform(low=0., high=0.2, size=self.img.shape)
            im_noise = cv2.add(self.image, uniform_array)
            print('uniform var 0.2')
        elif ('var0.6' in noise_type):
            uniform_array = np.random.uniform(low=0., high=0.6, size=self.img.shape)
            im_noise = cv2.add(self.image, uniform_array)
            print('uniform var 0.6')
        elif ('var1.2' in noise_type):
            uniform_array = np.random.uniform(low=0., high=1.2, size=self.img.shape)
            im_noise = cv2.add(self.image, uniform_array)
            print('uniform var 1.2')
        return im_noise


    def reconstruct(self, data, op, perc):

        """
        Convenience function to calculate reconstruction using top
        `perc` percent of coefficients of a given operator `op`.
         """
        y = op * data.ravel()
        denoise = np.zeros_like(y)
        # Order coefficients by strength
        strong_idx = np.argsort(-np.abs(y))
        strong = np.abs(y)[strong_idx]

        # Select only top `perc`% coefficients
        strong_idx = strong_idx[:int(np.rint(len(strong_idx) * perc))]
        denoise[strong_idx] = y[strong_idx]

        data_denoise = op.inverse(denoise).reshape(data.shape)
        return np.real(data_denoise), strong

if __name__=="__main__":
    noise = sys.argv[1]
    img_path = sys.argv[2]
    print(noise)
    fdct = FDCT(noise, img_path)
    #im = []
    perc = 0.5
    #img = cv2.imread("/home/mahesh/thesis/de-noise/tf-faster-rcnn/data/6thfloorData/6thFloorTest/JPEGImages/I004401.jpg")    
    if 'gaussian' in noise:
        im = fdct.add_gaussian_noise(noise)
        FDCT = cl.FDCT3D(im.shape, nbscales=4, nbangles_coarse=16)
        d_dct, dct_strong = fdct.reconstruct(im, FDCT, perc=perc)
        image =(d_dct * 255).astype(np.uint8)
        cv2.imwrite("temp.png", image)
    elif('poisson' in noise):
        im = fdct.add_poisson_noise(noise)
        FDCT = cl.FDCT3D(im.shape, nbscales=4, nbangles_coarse=16)
        d_dct, dct_strong = fdct.reconstruct(im, FDCT, perc=perc)
        image =(d_dct * 255).astype(np.uint8)
        cv2.imwrite("temp.png", image)
    elif('sap' in noise):
        im = fdct.add_sap_noise(noise)
        FDCT = cl.FDCT3D(im.shape, nbscales=4, nbangles_coarse=16)
        d_dct, dct_strong = fdct.reconstruct(im, FDCT, perc=perc)
        image =(d_dct * 255).astype(np.uint8)
        cv2.imwrite("temp.png", image)
    elif('speckle' in noise):
        im = fdct.add_speckle_noise(noise)
        FDCT = cl.FDCT3D(im.shape, nbscales=4, nbangles_coarse=16)
        d_dct, dct_strong = fdct.reconstruct(im, FDCT, perc=perc)
        image =(d_dct * 255).astype(np.uint8)
        cv2.imwrite("temp.png", image)
    elif('uniform' in noise):
        im = fdct.add_uniform_noise(noise)
        #perc = 0.1
        FDCT = cl.FDCT3D(im.shape, nbscales=4, nbangles_coarse=16)
        d_dct, dct_strong = fdct.reconstruct(im, FDCT, perc=perc)
        image =(d_dct * 255).astype(np.uint8)
        cv2.imwrite("temp.png", image)
    else:
        pass
        #d_dct, dct_strong = fdct.reconstruct(im, FDCT, perc=perc)
        #print(d_dct)
    #d_dct, dct_strong = fdct.reconstruct(self.imag3, FDCT, perc=perc)
    #c_img = (d_dct * 255).astype(np.uint8)
    #cv2.immwrite("test.png", f_img)
    #return c_img
