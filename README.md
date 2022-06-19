# Image denoising for Object detection with Faster R-CNN:
The objective of this project is to,
  - Implement Object detection with Faster R-CNN and fit custom dataset(RRLab data set.)
  - Introduce various image noise such as Gaussian noise, salt-and-pepper noise, speckle noise, periodic noise, quantization noise, Poisson noise, Brownian noise, Gamma and Rayleigh noise.
  - Introduce various denoising methods such as Gaussian blur, mean filter, median filter, Bilateral filter and Wavelet filter.
  - Validate the effect of various noise types with varying levels of intensities on object detection.
  - Cross validate the effect of various denoising methods for different noise types and combined noise types.
  - Infer the intrinsic dimension of data representation with TwoNN to understand the model's interpretability of noise and denoising methods.
  
# Faster R-CNN
A Tensorflow implementation of faster RCNN detection framework by [here](https://github.com/endernewton/tf-faster-rcnn). This repository is based on the python Caffe implementation of faster RCNN available [here](https://github.com/rbgirshick/py-faster-rcnn).

**Note**: Several minor modifications are made when reimplementing the framework, which give potential improvements. For details about the modifications and ablative analysis, please refer to the technical report [An Implementation of Faster RCNN with Study for Region Sampling](https://arxiv.org/pdf/1702.02138.pdf). For details about the faster RCNN architecture please refer to the paper [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](http://arxiv.org/pdf/1506.01497.pdf).

### Detection Performance
The current code supports **VGG16**, **Resnet V1**. We mainly tested it on plain Resnet101 architecture. As the baseline, we report numbers using a single model on a single convolution layer, so no multi-scale, no multi-stage bounding box regression, no skip-connection, no extra input is used.



With Resnet101 (last ``conv4``):
  - Train on VOC 2007 trainval and test on VOC 2007 test, **75.7**.
  - Train on VOC 2007+2012 trainval and test on VOC 2007 test (R-FCN schedule), **79.8**.
  - Train on COCO 2014 trainval35k and test on minival (900k/1190k), **35.4**.


**Note**:  
  - The images are injected with various types of noise at varying levels of intensities are are evaluated for object detection individually and also as a mixture of noise types. 
  - The noise are generated based on the respecitve probability density function with mean and variance. 
  - The Faster R-CNN implementation keeps the small proposals (\< 16 pixels width/height), which results in  good performance for small objects.
  - The threshold (instead of 0.05) is not set for a detection to be included in the final result, which increases recall.
  - Weight decay is set to 1e-4.
  - For other minor modifications, please check the [report](https://arxiv.org/pdf/1702.02138.pdf). Notable ones include using ``crop_and_resize``, and excluding ground truth boxes in RoIs during training.  
  - For Resnets, the first block (total 4) are fixed when fine-tuning the network, and only use ``crop_and_resize`` to resize the RoIs (7x7) without max-pool. The final feature maps are average-pooled for classification and regression. All batch normalization parameters are fixed. Learning rate for biases is not doubled.
  
  
![](data/imgs/gt.png)      |  ![](data/imgs/pred.png)
:-------------------------:|:-------------------------:
Displayed Ground Truth on Tensorboard |  Displayed Predictions on Tensorboard

### Prerequisites
  - A basic Tensorflow installation. The code follows **r1.2** format. If you are using r1.0, please check out the r1.0 branch to fix the slim Resnet block issue. If you are using an older version (r0.1-r0.12), please check out the r0.12 branch. While it is not required, for experimenting the original RoI pooling (which requires modification of the C++ code in tensorflow), you can check out this tensorflow [fork](https://github.com/endernewton/tensorflow) and look for ``tf.image.roi_pooling``.
  - Python packages you might not have: `cython`, `opencv-python`, `easydict` (similar to [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)). For `easydict` make sure you have the right version. This codebase use 1.6.  

### Installation
1. Clone the repository
  ```Shell
  git clone https://github.com/mahesh-kashyap-ml/image-denoising.git
  ```

2. Update your -arch in setup script to match your GPU
  ```Shell
  cd image-denoising/lib
  # Change the GPU architecture (-arch) if necessary
  vim setup.py
  ```

  | GPU model  | Architecture |
  | ------------- | ------------- |
  | TitanX (Maxwell/Pascal) | sm_52 |
  | GTX 960M | sm_50 |
  | GTX 1080 (Ti) | sm_61 |
  | Grid K520 (AWS g2.2xlarge) | sm_30 |
  | Tesla K80 (AWS p2.xlarge) | sm_37 |

 GPU based code (for NMS) will be used by default. So, to use CPU tensorflow, please set **USE_GPU_NMS False** to get the correct output.


3. Build the Cython modules
  ```Shell
  make clean
  make
  cd ..
  ```

### Setup data
Locate the RRLab dataset folder and create a soft link under ./data with the name **6thfloorData**

Pre-trained ImageNet models can be downloaded by  
```Shell
cd $FRCN_ROOT
./data/scripts/fetch_imagenet_models.sh
```

### Demo and Test with pre-trained models
1. Download pre-trained model or The pre-trained models are available in this directory. `/home/mahesh/thesis/de-noise/tf-faster-rcnn/output/res101/rrData_2021_train`
  

2. Create a folder and a soft link to use the pre-trained model
  ```Shell
  NET=res101
  TRAIN_IMDB=rrData
  mkdir -p output/${NET}/${TRAIN_IMDB}
  cd output/${NET}/${TRAIN_IMDB}
  ln -s ../../../data/6thfloorData ./default
  cd ../../..
  ```

3. The General syntax for training or testing with different noise types and denoise methods are as follows:
  - ./experiments/scripts/test_faster_rcnn.sh <arg1> <arg2> <arg3>
    - arg1 is the GPU_ID
    - arg2 is the backbone network. The options are res101, res50, vgg16. <Make sure to download the pretrained model of the respective network.>
    - arg3 is the noise and denoise type. The syntax for arg3 is **{noise type}_{denoise method}_var{noiselevel}**. To train or test with just noise type(without any denoising methods, skip the 'denoise method' )
    - The possible vaules for noise types are : gaussian, sap, speckle, poisson, quant, uniform, periodic, brownian, gamm, rayleigh.
    - The possible vaules for noise types are : gaus_blur, mean, median, bilateral, wavelet.
    - The possible values for various intensities (low, medium and high) are given in the below table.
      | Noise types | low | medium | high |
      | --- | --- | --- | --- |
      | gaussian | 0.1 | 1.0 | 1.5 |
      | sap | 0.2 | 0.4 | 0.8 |
      | speckle | 0.5 | 1.0 | 2.0 |
      | quant | 10 | 7 | 3 |
      | uniform | 0.2 | 0.6 | 1.2 |
      | brownian | 0.9 | 0.09 | 0.009 |
      | periodic | 3.14 | 100 | size |
      | gamma | 0.05 | 0.1 | 0.2 |
      | rayleigh | 0.1 | 0.2 | 0.3 |   
      
      - for example, gaussian_var0.1, gaussian_var1.0, gaussian_mean_var0.1, gaussian_wavelet_var1.5, 
    - The code supports to train and test with mixture of noise intensities. The intensity will be randomly chosen at runtime. 
      - use just the noise type: ./experiments/scripts/test_faster_rcnn.sh <arg1>
    - The code also supports training and testing with mixture of noise models at specific level of intensity and also with varying levels of intensities.
      - train and test with noise only; for arg3: noise_mix_var_all_low, noise_mix_var_all_medium, noise_mix_var_all_high and noise_mix_var_all
      - train and test with denoise method; for  arg3: noise_mix_var_all_{denoise method}
      
  
      
4. Test with pre-trained Resnet101 models
  ```Shell
  GPU_ID=0
  ./experiments/scripts/test_faster_rcnn.sh $GPU_ID rrData res101 gaussian
  ```
  
### Train your own model
1. Download pre-trained models and weights. The current code support VGG16 and Resnet V1 models. Pre-trained models are provided by slim, you can get the pre-trained models [here](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models) and set them in the ``data/imagenet_weights`` folder. For example for VGG16 model, you can set up like:
   ```Shell
   mkdir -p data/imagenet_weights
   cd data/imagenet_weights
   wget -v http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
   tar -xzvf vgg_16_2016_08_28.tar.gz
   mv vgg_16.ckpt vgg16.ckpt
   cd ../..
   ```
   For Resnet101, you can set up like:
   ```Shell
   mkdir -p data/imagenet_weights
   cd data/imagenet_weights
   wget -v http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz
   tar -xzvf resnet_v1_101_2016_08_28.tar.gz
   mv resnet_v1_101.ckpt res101.ckpt
   cd ../..
   ```

2. Train (and test, evaluation)
  ```Shell
  ./experiments/scripts/train_faster_rcnn.sh [GPU_ID] [DATASET] [NET] [NOISE_TYPE]
  # GPU_ID is the GPU you want to test on
  # NET in {vgg16, res50, res101, res152} is the network arch to use
  # DATASET {rrData is defined in train_faster_rcnn.sh
  # NOISE_TYPE {gaussian, sap, speckle, poisson, quant, uniform, periodic, brownian, gamm, rayleigh.}
  # Examples:
  ./experiments/scripts/train_faster_rcnn.sh 0 pascal_voc vgg16
  ./experiments/scripts/train_faster_rcnn.sh 1 coco res101
 
3. Visualization with Tensorboard
  ```Shell
  tensorboard --logdir=tensorboard/res101/rrData/ --port=7001 &  
  ```

4. Test and evaluate
  ```Shell
  ./experiments/scripts/test_faster_rcnn.sh [GPU_ID] [DATASET] [NET] [NOISE_TYPE]
  # GPU_ID is the GPU you want to test on
  # NET in {vgg16, res50, res101, res152} is the network arch to use
  # DATASET {rrData} is defined in test_faster_rcnn.sh
  # NOISE_TYPE {gaussian, sap, speckle, poisson, quant, uniform, periodic, brownian, gamm, rayleigh.}
  # Examples:
  ./experiments/scripts/test_faster_rcnn.sh 0 pascal_voc vgg16
  ./experiments/scripts/test_faster_rcnn.sh 1 coco res101
  ```
By default, trained networks are saved under:

```
output/[NET]/[DATASET]/[NOISE_TYPE]/
```

Tensorboard information for train and validation is saved under:

```
tensorboard/[NET]/[rrData_2021_train]/[NOISE_TYPE]/
tensorboard/[NET]/[rrData_2021_train]/[rrData_2021_train]_val/
```
