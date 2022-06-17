# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi He and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from tensorflow.contrib.slim.python.slim.nets.resnet_v1 import resnet_v1_block
import numpy as np

from nets.network import Network
from model.config import cfg

from scipy.stats import pearsonr
from sklearn import linear_model
from math import sqrt
from scipy.spatial.distance import pdist, squareform

def resnet_arg_scope(is_training=True,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
  batch_norm_params = {
    'is_training': False,
    'decay': batch_norm_decay,
    'epsilon': batch_norm_epsilon,
    'scale': batch_norm_scale,
    'trainable': False,
    'updates_collections': tf.GraphKeys.UPDATE_OPS
  }

  with arg_scope(
      [slim.conv2d],
      weights_regularizer=slim.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY),
      weights_initializer=slim.variance_scaling_initializer(),
      trainable=is_training,
      activation_fn=tf.nn.relu,
      normalizer_fn=slim.batch_norm,
      normalizer_params=batch_norm_params):
    with arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
      return arg_sc

class resnetv1(Network):
  def __init__(self, num_layers=50):
    Network.__init__(self)
    self._feat_stride = [16, ]
    self._feat_compress = [1. / float(self._feat_stride[0]), ]
    self._num_layers = num_layers
    self._scope = 'resnet_v1_%d' % num_layers
    self._decide_blocks()

  def _crop_pool_layer(self, bottom, rois, name):
    with tf.variable_scope(name) as scope:
      batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
      # Get the normalized coordinates of bboxes
      bottom_shape = tf.shape(bottom)
      height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self._feat_stride[0])
      width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self._feat_stride[0])
      x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
      y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
      x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
      y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
      # Won't be back-propagated to rois anyway, but to save time
      bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], 1))
      if cfg.RESNET.MAX_POOL:
        pre_pool_size = cfg.POOLING_SIZE * 2
        crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size],
                                         name="crops")
        crops = slim.max_pool2d(crops, [2, 2], padding='SAME')
      else:
        crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [cfg.POOLING_SIZE, cfg.POOLING_SIZE],
                                         name="crops")
    return crops

  # Do the first few layers manually, because 'SAME' padding can behave inconsistently
  # for images of different sizes: sometimes 0, sometimes 1
  def _build_base(self):
    with tf.variable_scope(self._scope, self._scope):
      net = resnet_utils.conv2d_same(self._image, 64, 7, stride=2, scope='conv1')
      net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
      net = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='pool1')

    return net

  def _image_to_head(self, is_training, reuse=None):
    assert (0 <= cfg.RESNET.FIXED_BLOCKS <= 3)
    # Now the base is always fixed during training
    with slim.arg_scope(resnet_arg_scope(is_training=False)):
      net_conv = self._build_base()
    if cfg.RESNET.FIXED_BLOCKS > 0:
      with slim.arg_scope(resnet_arg_scope(is_training=False)):
        net_conv, _ = resnet_v1.resnet_v1(net_conv,
                                           self._blocks[0:cfg.RESNET.FIXED_BLOCKS],
                                           global_pool=False,
                                           include_root_block=False,
                                           reuse=reuse,
                                           scope=self._scope)
    if cfg.RESNET.FIXED_BLOCKS < 3:
      with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
        net_conv, _ = resnet_v1.resnet_v1(net_conv,
                                           self._blocks[cfg.RESNET.FIXED_BLOCKS:-1],
                                           global_pool=False,
                                           include_root_block=False,
                                           reuse=reuse,
                                           scope=self._scope)

    self._act_summaries.append(net_conv)
    self._layers['head'] = net_conv
    self._layers['end_points'] = _
    return net_conv

  def _head_to_tail(self, pool5, is_training, reuse=None):
    with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
      fc7, _ = resnet_v1.resnet_v1(pool5,
                                   self._blocks[-1:],
                                   global_pool=False,
                                   include_root_block=False,
                                   reuse=reuse,
                                   scope=self._scope)
      # average pooling done by reduce_mean
      fc7 = tf.reduce_mean(fc7, axis=[1, 2])
    self._layers['fc7'] = fc7
    self._layers['fc7_end_points'] = _
    return fc7

  def _decide_blocks(self):
    # choose different blocks for different number of layers
    if self._num_layers == 50:
      self._blocks = [resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
                      resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
                      # use stride 1 for the last conv4 layer
                      resnet_v1_block('block3', base_depth=256, num_units=6, stride=1),
                      resnet_v1_block('block4', base_depth=512, num_units=3, stride=1)]

    elif self._num_layers == 101:
      self._blocks = [resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
                      resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
                      # use stride 1 for the last conv4 layer
                      resnet_v1_block('block3', base_depth=256, num_units=23, stride=1),
                      resnet_v1_block('block4', base_depth=512, num_units=3, stride=1)]

    elif self._num_layers == 152:
      self._blocks = [resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
                      resnet_v1_block('block2', base_depth=128, num_units=8, stride=2),
                      # use stride 1 for the last conv4 layer
                      resnet_v1_block('block3', base_depth=256, num_units=36, stride=1),
                      resnet_v1_block('block4', base_depth=512, num_units=3, stride=1)]

    else:
      # other numbers are not supported
      raise NotImplementedError

  def get_variables_to_restore(self, variables, var_keep_dic):
    variables_to_restore = []

    for v in variables:
      # exclude the first conv layer to swap RGB to BGR
      if v.name == (self._scope + '/conv1/weights:0'):
        self._variables_to_fix[v.name] = v
        continue
      if v.name.split(':')[0] in var_keep_dic:
        print('Variables restored: %s' % v.name)
        variables_to_restore.append(v)

    return variables_to_restore

  def fix_variables(self, sess, pretrained_model):
    print('Fix Resnet V1 layers..')
    with tf.variable_scope('Fix_Resnet_V1') as scope:
      with tf.device("/cpu:0"):
        # fix RGB to BGR
        conv1_rgb = tf.get_variable("conv1_rgb", [7, 7, 3, 64], trainable=False)
        restorer_fc = tf.train.Saver({self._scope + "/conv1/weights": conv1_rgb})
        restorer_fc.restore(sess, pretrained_model)

        sess.run(tf.assign(self._variables_to_fix[self._scope + '/conv1/weights:0'], 
                           tf.reverse(conv1_rgb, [2])))

  def estimate(self, X, fraction=0.9,verbose=False):    
    """
    Returns:            
    x : log(mu)    
    y : -(1-F(mu)) 
    reg : linear regression y ~ x structure obtained with scipy.stats.linregress
    (reg.slope is the intrinsic dimension estimate)
    r : determination coefficient of y ~ x
    pval : p-value of y ~ x"""
     
    # sort distance matrix
    Y = np.sort(X,axis=1,kind='quicksort')
    print(Y)

    # clean data
    k1 = Y[:,1]
    k2 = Y[:,2]



    zeros = np.where(k1 == 0)[0]
    if verbose:
        print('Found n. {} elements for which r1 = 0'.format(zeros.shape[0]))
        print(zeros)

    degeneracies = np.where(k1 == k2)[0]
    if verbose:
        print('Found n. {} elements for which r1 = r2'.format(degeneracies.shape[0]))
        print(degeneracies)

    good = np.setdiff1d(np.arange(Y.shape[0]), np.array(zeros) )
    good = np.setdiff1d(good,np.array(degeneracies))
    
    #print("good: ", good)

    if verbose:
        print('Fraction good points: {}'.format(good.shape[0]/Y.shape[0]))
    
    k1 = k1[good]
    k2 = k2[good]    

    
    # n.of points to consider for the linear regression, fraction of the data considered for dimensionality estimation(default is 0.9)
    npoints = int(np.floor(good.shape[0]*fraction))


    # define mu and Femp
    N = good.shape[0]
    mu = np.sort(np.divide(k2, k1), axis=None,kind='quicksort')
    #print("mu: ", mu)
    Femp = (np.arange(1,N+1,dtype=np.float64) )/N
    
    # take logs (leave out the last element because 1-Femp is zero there)
    x = np.log(mu[:-2])
    y = -np.log(1 - Femp[:-2])

    # regression
    regr = linear_model.LinearRegression(fit_intercept=False)
    regr.fit(x[0:npoints,np.newaxis],y[0:npoints,np.newaxis]) 
    r,pval = pearsonr(x[0:npoints], y[0:npoints]) 
    print("x: {}, y: {}, regr_coef: {}, r: {}, pval:{}".format(x,y,regr.coef_[0][0],r,pval))
    return x,y,regr.coef_[0][0],r,pval

  def computeID(self, r, nres,fraction):
        verbose = False
        method = 'euclidean'
        fraction = 0.9
        nres = 20
        ID = []
        n = int(np.round(r.shape[0]*fraction))
        # squareform: convert a vector-form distance vector to square-form distance martix.
            #pdist: Pairwise distance between observations in n-dimensional space.
        dist = squareform(pdist(r, 'euclidean'))
        for i in range(nres):
            dist_s = dist
            perm = np.random.permutation(dist.shape[0])[0:n]
            dist_s = dist_s[perm,:]
            dist_s = dist_s[:,perm]
            ID.append(self.estimate(dist_s,verbose=verbose)[2])
        mean = np.mean(ID) 
        error = np.std(ID) 
        return mean,error      

  def model_summary(self, sess, blobs, iter, max_iters, train_op):
    feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'], 
                 self._gt_boxes: blobs['gt_boxes']}
    """feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                 }"""
    model_vars = tf.trainable_variables()
    #model_all_vars = tf.get_model_variables()
    #slim.model_analyzer.analyze_vars(model_vars, print_info=True)
    #conv = self._image_to_head(is_training=True, reuse=None)
    #print(conv)
    feat11 = np.array([], dtype=np.float32)
    #feat22 = np.array([])
    #feat33 = np.array([])
    #feat44 = np.array([])
    #feat55 = np.array([])
    #feat66 = np.array([])
    #feat77 = np.array([])
    #feat88 = np.array([])
    if iter < max_iters:
        if iter == 0:
            feat1 = sess.run(self._layers['end_points']['resnet_v1_101/block2/unit_1/bottleneck_v1/conv1'], feed_dict=feed_dict)
            feat1a = feat1.reshape(feat1.shape[0], -1)
            feat11.append(feat1a)
        else:
            feat1 = sess.run(self._layers['end_points']['resnet_v1_101/block2/unit_1/bottleneck_v1/conv1'], feed_dict=feed_dict)
            feat11 = tf.concat([feat11, (feat1.reshape(feat1.shape[0], -1))], 0)

        #feat1 = sess.run(self._layers['end_points']['resnet_v1_101/block2/unit_1/bottleneck_v1/conv1'], feed_dict=feed_dict)
        #feat11 = tf.reshape(tf.constant([], dtype=tf.float32), (feat1.reshape(feat1.shape[0], -1)))
        #feat11 = tf.placeholder(tf.float32, (feat1.reshape(feat1.shape[0], -1)))
        #feat11 = tf.concat([(feat1.reshape(feat1.shape[0], -1)), (feat1.reshape(feat1.shape[0], -1))], 0)
        #feat1 = tf.concat([feat1, (feat1.reshape(feat1.shape[0], -1))], 0)
        #print(feat11.shape)
        """feat2 = sess.run(self._layers['end_points']['resnet_v1_101/block2/unit_1/bottleneck_v1/conv2'], feed_dict=feed_dict)
        feat22 = tf.concat([(feat2.reshape(feat2.shape[0], -1)), (feat2.reshape(feat2.shape[0], -1))], 0)
        print(feat22.shape)
        feat3 = sess.run(self._layers['end_points']['resnet_v1_101/block2/unit_1/bottleneck_v1/conv3'], feed_dict=feed_dict)
        feat33 = tf.concat([(feat3.reshape(feat3.shape[0], -1)), (feat3.reshape(feat3.shape[0], -1))], 0)
        feat4 = sess.run(self._layers['end_points']['resnet_v1_101/block2/unit_2/bottleneck_v1/conv1'], feed_dict=feed_dict)
        feat44 = tf.concat([(feat4.reshape(feat4.shape[0], -1)), (feat4.reshape(feat4.shape[0], -1))], 0)
        feat5 = sess.run(self._layers['end_points']['resnet_v1_101/block2/unit_2/bottleneck_v1/conv2'], feed_dict=feed_dict)
        feat55 = tf.concat([(feat5.reshape(feat5.shape[0], -1)), (feat5.reshape(feat5.shape[0], -1))], 0)
        feat6 = sess.run(self._layers['end_points']['resnet_v1_101/block2/unit_2/bottleneck_v1/conv3'], feed_dict=feed_dict)
        feat66 = tf.concat([(feat6.reshape(feat6.shape[0], -1)), (feat6.reshape(feat6.shape[0], -1))], 0)
        feat7 = sess.run(self._layers['end_points']['resnet_v1_101/block2/unit_3/bottleneck_v1/conv1'], feed_dict=feed_dict)
        feat77 = tf.concat([(feat7.reshape(feat7.shape[0], -1)), (feat7.reshape(feat7.shape[0], -1))], 0)
        feat8 = sess.run(self._layers['end_points']['resnet_v1_101/block3/unit_1/bottleneck_v1/conv1'], feed_dict=feed_dict)
        feat88 = tf.concat([(feat8.reshape(feat8.shape[0], -1)), (feat8.reshape(feat8.shape[0], -1))], 0)
    elif iter == max_iters:
        print(feat11.shape)
        print(feat22.shape)
        print(feat3.shape)
        print(feat4.shape)
        print(feat5.shape)
        print(feat6.shape)
        print(feat7.shape)
        print(feat8.shape)"""
    #print(feat1)
    #print("conv1")
    #print(self.computeID(feat1,20,0.9))
    #feat2 = sess.run(self._layers['end_points']['resnet_v1_101/block2/unit_1/bottleneck_v1/conv2'], feed_dict=feed_dict)
    #print(feat2)
    #print("conv2")
    #slim.model_analyzer.analyze_vars(model_all_vars, print_info=True)
    #print(self._layers['end_points']['resnet_v1_101/block2/unit_1/bottleneck_v1/conv1'])
    #print(self._layers[''])
    #print(tf.get_default_graph().get_tensor_by_name('resnet_v1_101/block2/unit_1/bottleneck_v1/conv1/weights:0'))




