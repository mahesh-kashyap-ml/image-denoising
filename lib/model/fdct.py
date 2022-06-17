import curvelops as cl
import cv2
from skimage import img_as_float
import numpy as np
from skimage.util import random_noise
import sys
from sklearn.cluster import MiniBatchKMeans
from scipy.stats import uniform
from scipy.stats import gamma
from scipy.stats import rayleigh
from PIL import Image
from PIL import ImageEnhance


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
    
    def add_quant_noise(self, noise_type):
        im_noise = np.array([])
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB)
        h, w = img.shape[:2]
        #clor quantization, using K-Means clustering.
        #Usually this noise is found while converting analog to digital, or continuous random variable to discreate.
        image = img.reshape((img.shape[0] * img.shape[1], 3))
        if ('var3' in noise_type):
            clt = MiniBatchKMeans(n_clusters= 3)
            labels = clt.fit_predict(image)
            quant = clt.cluster_centers_.astype("uint8")[labels]
            quant = quant.reshape(h, w, 3)
            im_noise = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
            print('quantization with cluster 3')
        elif ('var7' in noise_type):
            clt = MiniBatchKMeans(n_clusters= 7)
            labels = clt.fit_predict(image)
            quant = clt.cluster_centers_.astype("uint8")[labels]
            quant = quant.reshape(h, w, 3)
            im_noise = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
            print('quantization with cluster 7')
        elif ('var10' in noise_type):
            clt = MiniBatchKMeans(n_clusters= 10)
            labels = clt.fit_predict(image)
            quant = clt.cluster_centers_.astype("uint8")[labels]
            quant = quant.reshape(h, w, 3)
            im_noise = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
            print('quantization with cluster 10')
        return im_noise
    
    
    def add_brownian_noise(self, noise_type):
        im_noise = np.array([])
        h, w = self.img.shape[:2]
        n=self.img.size
        if ('var0.9' in noise_type):
            dt = 0.9
            dB = np.sqrt(dt) * np.random.normal(size=(n-1,))
            #brownian motion starts at zero
            B0 = np.zeros(shape=(1,))
            #brownian motion is to concatenate the intial value with the cumulative sum of the increments.
            B = np.concatenate((B0, np.cumsum(dB)))
            brownian = (B * 255).astype(np.uint8)
            brownian_noise = brownian.reshape(h,w,3)
            im_noise = cv2.add(self.img, brownian_noise)
            print('brownian var 0.9')
        elif ('var0.09' in noise_type):
            dt = 0.09
            dB = np.sqrt(dt) * np.random.normal(size=(n-1,))
            B0 = np.zeros(shape=(1,))
            B = np.concatenate((B0, np.cumsum(dB)))
            brownian = (B * 255).astype(np.uint8)
            brownian_noise = brownian.reshape(h,w,3)
            im_noise = cv2.add(self.img, brownian_noise)
            print('brownian var 0.09')
        elif ('var0.009' in noise_type):
            dt = 0.009
            dB = np.sqrt(dt) * np.random.normal(size=(n-1,))
            B0 = np.zeros(shape=(1,))
            B = np.concatenate((B0, np.cumsum(dB)))
            brownian = (B * 255).astype(np.uint8)
            brownian_noise = brownian.reshape(h,w,3)
            im_noise = cv2.add(self.img, brownian_noise)
            print('brownian var 0.009')
        return im_noise

    def add_periodic_noise(self, noise_type):
        im_noise = np.array([])
        h, w = self.img.shape[:2]
        size = self.img.size
        if ('var3.14' in noise_type):
            time = (np.linspace(-np.pi, np.pi, size))
            amplitude = np.sin(time)
            periodic_array = (amplitude * 255).astype(np.uint8)
            periodic_noise = periodic_array.reshape(h,w,3)
            im_noise = cv2.add(self.img, periodic_noise)
            print('periodic amplitude pi')
        elif ('var100' in noise_type):
            time = (np.linspace(-100, 100, size))
            amplitude = np.sin(time)
            periodic_array = (amplitude * 255).astype(np.uint8)
            periodic_noise = periodic_array.reshape(h,w,3)
            im_noise = cv2.add(self.img, periodic_noise)
            print('periodic amplitude 100')
        elif ('varsize' in noise_type):
            time = (np.linspace(-size, size, size))
            amplitude = np.sin(time)
            periodic_array = (amplitude * 255).astype(np.uint8)
            periodic_noise = periodic_array.reshape(h,w,3)
            im_noise = cv2.add(self.img, periodic_noise)
            print('periodic amplitude size')
        return im_noise
    
    def add_rayleigh_noise(self, noise_type):
        im_noise = np.array([])        
        image = img_as_float(self.img)
        if ('var0.1' in noise_type):
            rayleigh_dist = rayleigh.rvs(loc=0., scale=0.1, size=image.shape)
            rayleigh_array = cv2.add(image, rayleigh_dist)
            im_noise = (rayleigh_array * 255).astype(np.uint8)
            print('rayleigh var 0.1')
        elif ('var0.2' in noise_type):
            rayleigh_dist = rayleigh.rvs(loc=0., scale=0.2, size=image.shape)
            rayleigh_array = cv2.add(image, rayleigh_dist)
            im_noise = (rayleigh_array * 255).astype(np.uint8)
            print('rayleigh var 0.2')
        elif ('var0.3' in noise_type):
            rayleigh_dist = rayleigh.rvs(loc=0., scale=0.3, size=image.shape)
            rayleigh_array = cv2.add(image, rayleigh_dist)
            im_noise = (rayleigh_array * 255).astype(np.uint8)
            print('rayleigh var 0.3')
        return im_noise

    def add_shader(self, img_path):
        factor = 3
        im = Image.open(img_path)
        im_out = ImageEnhance.Brightness(im).enhance(factor)
        im_noise = np.array(im_out)
        print('PIL enhance')
        return im_noise

    def retain_original(self, img_path):
        im_noise = cv2.imread(img_path)
        #introduce gaussian noise.
        print('original')
        return im_noise


    def add_gamma_noise(self, noise_type):
        im_noise = np.array([])
        image = img_as_float(self.img)
        a = 1.99
        if ('var0.1' in noise_type):
            gamma_dist = gamma.rvs(a, loc=0., scale=0.1, size=image.shape)
            gamma_array = cv2.add(image, gamma_dist)
            im_noise = (gamma_array * 255).astype(np.uint8)
            print('gamma var 0.1')            
        elif ('var0.3' in noise_type):
            gamma_dist = gamma.rvs(a, loc=0., scale=0.3, size=image.shape)
            gamma_array = cv2.add(image, gamma_dist)
            im_noise = (gamma_array * 255).astype(np.uint8)
            print('gamma var 0.3')
        elif ('var0.7' in noise_type):
            gamma_dist = gamma.rvs(a, loc=0., scale=0.7, size=image.shape)
            gamma_array = cv2.add(image, gamma_dist)
            im_noise = (gamma_array * 255).astype(np.uint8)
            print('gamma var 0.7')
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
    elif('quant' in noise):
        im = fdct.add_quant_noise(noise)
        #perc = 0.1
        FDCT = cl.FDCT3D(im.shape, nbscales=4, nbangles_coarse=16)
        d_dct, dct_strong = fdct.reconstruct(im, FDCT, perc=perc)
        image =(d_dct * 255).astype(np.uint8)
        cv2.imwrite("temp.png", image)
    elif('periodic' in noise):
        im = fdct.add_periodic_noise(noise)
        #perc = 0.1
        FDCT = cl.FDCT3D(im.shape, nbscales=4, nbangles_coarse=16)
        d_dct, dct_strong = fdct.reconstruct(im, FDCT, perc=perc)
        image =(d_dct * 255).astype(np.uint8)
        cv2.imwrite("temp.png", image)
    elif('brownian' in noise):
        im = fdct.add_brownian_noise(noise)
        #perc = 0.1
        FDCT = cl.FDCT3D(im.shape, nbscales=4, nbangles_coarse=16)
        d_dct, dct_strong = fdct.reconstruct(im, FDCT, perc=perc)
        image =(d_dct * 255).astype(np.uint8)
        cv2.imwrite("temp.png", image)
    elif('gamma' in noise):
        im = fdct.add_gamma_noise(noise)
        #perc = 0.1
        FDCT = cl.FDCT3D(im.shape, nbscales=4, nbangles_coarse=16)
        d_dct, dct_strong = fdct.reconstruct(im, FDCT, perc=perc)
        image =(d_dct * 255).astype(np.uint8)
        cv2.imwrite("temp.png", image)
    elif('rayleigh' in noise):
        im = fdct.add_rayleigh_noise(noise)
        #perc = 0.1
        FDCT = cl.FDCT3D(im.shape, nbscales=4, nbangles_coarse=16)
        d_dct, dct_strong = fdct.reconstruct(im, FDCT, perc=perc)
        image =(d_dct * 255).astype(np.uint8)
        cv2.imwrite("temp.png", image)
    elif('shader' in noise):
        im = fdct.add_shader(img_path)
        #perc = 0.1
        FDCT = cl.FDCT3D(im.shape, nbscales=4, nbangles_coarse=16)
        d_dct, dct_strong = fdct.reconstruct(im, FDCT, perc=perc)
        image =(d_dct * 255).astype(np.uint8)
        cv2.imwrite("temp.png", image)
    elif('original' in noise):
        im = fdct.retain_original(img_path)
        #perc = 0.1
        FDCT = cl.FDCT3D(im.shape, nbscales=4, nbangles_coarse=16)
        d_dct, dct_strong = fdct.reconstruct(im, FDCT, perc=perc)
        image =(d_dct * 255).astype(np.uint8)
        cv2.imwrite("temp.png", image)    
    else:
        ("noise type definition not found!")

