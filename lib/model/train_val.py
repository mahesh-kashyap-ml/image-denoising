# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen and Zheqi He
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.config import cfg
import roi_data_layer.roidb as rdl_roidb
from roi_data_layer.layer import RoIDataLayer
from utils.timer import Timer
try:
  import cPickle as pickle
except ImportError:
  import pickle
import numpy as np
import os
import sys
import glob
import time
import wandb
import Automold as am
import Helpers as hp


os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

class SolverWrapper(object):
  """
    A wrapper class for the training process
  """

  def __init__(self, sess, network, imdb, roidb, valroidb, output_dir, tbdir, pretrained_model=None):
    self.net = network
    self.imdb = imdb
    self.roidb = roidb
    self.valroidb = valroidb
    self.output_dir = output_dir
    self.tbdir = tbdir
    # Simply put '_val' at the end to save the summaries from the validation set
    self.tbvaldir = tbdir + '_val'
    if not os.path.exists(self.tbvaldir):
      os.makedirs(self.tbvaldir)
    self.pretrained_model = pretrained_model

  def snapshot(self, sess, iter):
    net = self.net

    if not os.path.exists(self.output_dir):
      os.makedirs(self.output_dir)

    # Store the model snapshot
    filename = cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}'.format(iter) + '.ckpt'
    filename = os.path.join(self.output_dir, filename)
    self.saver.save(sess, filename)
    print('Wrote snapshot to: {:s}'.format(filename))

    # Also store some meta information, random state, etc.
    nfilename = cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}'.format(iter) + '.pkl'
    nfilename = os.path.join(self.output_dir, nfilename)
    # current state of numpy random
    st0 = np.random.get_state()
    # current position in the database
    cur = self.data_layer._cur
    # current shuffled indexes of the database
    perm = self.data_layer._perm
    # current position in the validation database
    cur_val = self.data_layer_val._cur
    # current shuffled indexes of the validation database
    perm_val = self.data_layer_val._perm

    # Dump the meta info
    with open(nfilename, 'wb') as fid:
      pickle.dump(st0, fid, pickle.HIGHEST_PROTOCOL)
      pickle.dump(cur, fid, pickle.HIGHEST_PROTOCOL)
      pickle.dump(perm, fid, pickle.HIGHEST_PROTOCOL)
      pickle.dump(cur_val, fid, pickle.HIGHEST_PROTOCOL)
      pickle.dump(perm_val, fid, pickle.HIGHEST_PROTOCOL)
      pickle.dump(iter, fid, pickle.HIGHEST_PROTOCOL)

    return filename, nfilename

  def from_snapshot(self, sess, sfile, nfile):
    print('Restoring model snapshots from {:s}'.format(sfile))
    self.saver.restore(sess, sfile)
    print('Restored.')
    # Needs to restore the other hyper-parameters/states for training, (TODO xinlei) I have
    # tried my best to find the random states so that it can be recovered exactly
    # However the Tensorflow state is currently not available
    with open(nfile, 'rb') as fid:
      st0 = pickle.load(fid)
      cur = pickle.load(fid)
      perm = pickle.load(fid)
      cur_val = pickle.load(fid)
      perm_val = pickle.load(fid)
      last_snapshot_iter = pickle.load(fid)

      np.random.set_state(st0)
      self.data_layer._cur = cur
      self.data_layer._perm = perm
      self.data_layer_val._cur = cur_val
      self.data_layer_val._perm = perm_val

    return last_snapshot_iter

  def get_variables_in_checkpoint_file(self, file_name):
    try:
      reader = pywrap_tensorflow.NewCheckpointReader(file_name)
      var_to_shape_map = reader.get_variable_to_shape_map()
      return var_to_shape_map 
    except Exception as e:  # pylint: disable=broad-except
      print(str(e))
      if "corrupted compressed block contents" in str(e):
        print("It's likely that your checkpoint file has been compressed "
              "with SNAPPY.")

  def construct_graph(self, sess):
    with sess.graph.as_default():
      # Set the random seed for tensorflow
      tf.set_random_seed(cfg.RNG_SEED)
      # Build the main computation graph
      layers = self.net.create_architecture('TRAIN', self.imdb.num_classes, tag='default',
                                            anchor_scales=cfg.ANCHOR_SCALES,
                                            anchor_ratios=cfg.ANCHOR_RATIOS)
      #print(layers)
      # Define the loss
      loss = layers['total_loss']
      # Set learning rate and momentum
      lr = tf.Variable(cfg.TRAIN.LEARNING_RATE, trainable=False)
      self.optimizer = tf.train.MomentumOptimizer(lr, cfg.TRAIN.MOMENTUM)

      # Compute the gradients with regard to the loss
      gvs = self.optimizer.compute_gradients(loss)
      # Double the gradient of the bias if set
      if cfg.TRAIN.DOUBLE_BIAS:
        final_gvs = []
        with tf.variable_scope('Gradient_Mult') as scope:
          for grad, var in gvs:
            scale = 1.
            if cfg.TRAIN.DOUBLE_BIAS and '/biases:' in var.name:
              scale *= 2.
            if not np.allclose(scale, 1.0):
              grad = tf.multiply(grad, scale)
            final_gvs.append((grad, var))
        train_op = self.optimizer.apply_gradients(final_gvs)
      else:
        train_op = self.optimizer.apply_gradients(gvs)

      # We will handle the snapshots ourselves
      self.saver = tf.train.Saver(max_to_keep=100000)
      # Write the train and validation information to tensorboard
      self.writer = tf.summary.FileWriter(self.tbdir, sess.graph)
      self.valwriter = tf.summary.FileWriter(self.tbvaldir)

      #write the wandb logs
      #wandb.tensorflow.log(tf.summary.merge_all())
      #wandb.init()
      #wandb.config.update(loss,lr,self.optimizer)

    return lr, train_op

  def find_previous(self):
    sfiles = os.path.join(self.output_dir, cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_*.ckpt.meta')
    sfiles = glob.glob(sfiles)
    sfiles.sort(key=os.path.getmtime)
    # Get the snapshot name in TensorFlow
    redfiles = []
    for stepsize in cfg.TRAIN.STEPSIZE:
      redfiles.append(os.path.join(self.output_dir, 
                      cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}.ckpt.meta'.format(stepsize+1)))
    sfiles = [ss.replace('.meta', '') for ss in sfiles if ss not in redfiles]

    nfiles = os.path.join(self.output_dir, cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_*.pkl')
    nfiles = glob.glob(nfiles)
    nfiles.sort(key=os.path.getmtime)
    redfiles = [redfile.replace('.ckpt.meta', '.pkl') for redfile in redfiles]
    nfiles = [nn for nn in nfiles if nn not in redfiles]

    lsf = len(sfiles)
    assert len(nfiles) == lsf

    return lsf, nfiles, sfiles

  def initialize(self, sess):
    # Initial file lists are empty
    np_paths = []
    ss_paths = []
    # Fresh train directly from ImageNet weights
    print('Loading initial model weights from {:s}'.format(self.pretrained_model))
    variables = tf.global_variables()
    # Initialize all variables first
    sess.run(tf.variables_initializer(variables, name='init'))
    var_keep_dic = self.get_variables_in_checkpoint_file(self.pretrained_model)
    # Get the variables to restore, ignoring the variables to fix
    variables_to_restore = self.net.get_variables_to_restore(variables, var_keep_dic)

    restorer = tf.train.Saver(variables_to_restore)
    restorer.restore(sess, self.pretrained_model)
    print('Loaded.')
    # Need to fix the variables before loading, so that the RGB weights are changed to BGR
    # For VGG16 it also changes the convolutional weights fc6 and fc7 to
    # fully connected weights
    self.net.fix_variables(sess, self.pretrained_model)
    print('Fixed.')
    last_snapshot_iter = 0
    rate = cfg.TRAIN.LEARNING_RATE
    stepsizes = list(cfg.TRAIN.STEPSIZE)

    return rate, last_snapshot_iter, stepsizes, np_paths, ss_paths

  def restore(self, sess, sfile, nfile):
    # Get the most recent snapshot and restore
    np_paths = [nfile]
    ss_paths = [sfile]
    # Restore model from snapshots
    last_snapshot_iter = self.from_snapshot(sess, sfile, nfile)
    # Set the learning rate
    rate = cfg.TRAIN.LEARNING_RATE
    stepsizes = []
    for stepsize in cfg.TRAIN.STEPSIZE:
      if last_snapshot_iter > stepsize:
        rate *= cfg.TRAIN.GAMMA
      else:
        stepsizes.append(stepsize)

    return rate, last_snapshot_iter, stepsizes, np_paths, ss_paths

  def remove_snapshot(self, np_paths, ss_paths):
    to_remove = len(np_paths) - cfg.TRAIN.SNAPSHOT_KEPT
    for c in range(to_remove):
      nfile = np_paths[0]
      os.remove(str(nfile))
      np_paths.remove(nfile)

    to_remove = len(ss_paths) - cfg.TRAIN.SNAPSHOT_KEPT
    for c in range(to_remove):
      sfile = ss_paths[0]
      # To make the code compatible to earlier versions of Tensorflow,
      # where the naming tradition for checkpoints are different
      if os.path.exists(str(sfile)):
        os.remove(str(sfile))
      else:
        os.remove(str(sfile + '.data-00000-of-00001'))
        os.remove(str(sfile + '.index'))
      sfile_meta = sfile + '.meta'
      os.remove(str(sfile_meta))
      ss_paths.remove(sfile)

  def train_model(self, sess, max_iters):
    # Build data layers for both training and validation set
    self.data_layer = RoIDataLayer(self.roidb, self.imdb.num_classes)
    self.data_layer_val = RoIDataLayer(self.valroidb, self.imdb.num_classes, random=True)


    # Construct the computation graph
    lr, train_op = self.construct_graph(sess)

    # Find previous snapshots if there is any to restore from
    lsf, nfiles, sfiles = self.find_previous()

    #initialize wandb
    #wandb.init(project='faster-rcnn-denoise', entity='mkashyap')
    #wandb.init(config=tf.flags.FLAGS, sync_tensorboard=True)
    wandb.init(project='tf-faster-rcnn-vgg', entity='mkashyap')
    #config = wandb.config
    #config.learning_rate = lr
    #config.train_op = train_op

    #wandb.config.update()

    #wandb.log({"custom": noise_mix_var_medium})
    
    #print(self.net.summary())
    # Initialize the variables or restore them from the last snapshot
    if lsf == 0:
      rate, last_snapshot_iter, stepsizes, np_paths, ss_paths = self.initialize(sess)
    else:
      rate, last_snapshot_iter, stepsizes, np_paths, ss_paths = self.restore(sess, 
                                                                            str(sfiles[-1]), 
                                                                            str(nfiles[-1]))
    #print(self.net.model_summary())
    timer = Timer()
    iter = last_snapshot_iter + 1
    last_summary_time = time.time()
    # Make sure the lists are not empty
    stepsizes.append(max_iters)
    stepsizes.reverse()
    next_stepsize = stepsizes.pop()
    while iter < max_iters + 1:
      # Learning rate
      if iter == next_stepsize + 1:
        # Add snapshot here before reducing the learning rate
        self.snapshot(sess, iter)
        rate *= cfg.TRAIN.GAMMA
        sess.run(tf.assign(lr, rate))
        next_stepsize = stepsizes.pop()
        config.learning_rate = rate

      timer.tic()
      # Get training data, one batch at a time
      blobs = self.data_layer.forward()




      now = time.time()
      if iter == 1 or now - last_summary_time > cfg.TRAIN.SUMMARY_INTERVAL:
        # Compute the graph with summary
        rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, total_loss, summary = \
          self.net.train_step_with_summary(sess, blobs, train_op)
        self.writer.add_summary(summary, float(iter))
        # Also check the summary on the validation set
        blobs_val = self.data_layer_val.forward()
        summary_val = self.net.get_summary(sess, blobs_val)
        self.valwriter.add_summary(summary_val, float(iter))
        last_summary_time = now
        #wandb.log({'iteration': iter, 'loss': total_loss})
        #config = wandb.config
        #config.learning_rate = rate
        #config.summary = summary
        wandb.log({'rpn_loss_cls': rpn_loss_cls, 'total_loss': total_loss})
        wandb.tensorflow.log(summary)
        #print(type(summary))
        #estimator.train(hooks=[wandb.tensorflow.WandbHook(steps_per_log=1000)])
        #print(type(summary))
        #ouput1 = self.net.model_summary(sess, blobs, iter, max_iters, train_op)
        """feed_dict = {self.net._image: blobs['data'], self.net._im_info: blobs['im_info'],
                 self.net._gt_boxes: blobs['gt_boxes']}
        feat1 = sess.run(self.net._layers['end_points']['resnet_v1_101/block2/unit_1/bottleneck_v1/conv1'], feed_dict=feed_dict)
        #feat11 = feat1.reshape(feat1.shape[0], -1)
        feat11 = feat1.view().reshape(feat1.shape[0], -1)
        print(feat11)
        feat2 = sesss.run(self.net._layers['end_points']['resnet_v1_101/block2/unit_1/bottleneck_v1/conv2'], feed_dict=feed_dict)
        #feat22 = feat2.reshape(feat2.shape[0], -1)
        feat22 = feat2.view().reshape(feat2.shape[0], -1)
        feat3 = sess.run(self.net._layers['end_points']['resnet_v1_101/block2/unit_1/bottleneck_v1/conv3'], feed_dict=feed_dict)
        #feat33 = feat3.reshape(feat3.shape[0], -1)
        feat33 = feat3.view().reshape(feat3.shape[0], -1)
        feat4 = sess.run(self.net._layers['end_points']['resnet_v1_101/block2/unit_2/bottleneck_v1/conv1'], feed_dict=feed_dict)
        feat44 = feat4.view().reshape(feat4.shape[0], -1)
        feat5 = sess.run(self.net._layers['end_points']['resnet_v1_101/block2/unit_2/bottleneck_v1/conv2'], feed_dict=feed_dict)
        feat55 = feat5.view().reshape(feat5.shape[0], -1)
        feat6 = sess.run(self.net._layers['end_points']['resnet_v1_101/block2/unit_2/bottleneck_v1/conv3'], feed_dict=feed_dict)
        feat66 = feat6.view().reshape(feat6.shape[0], -1)
        feat7 = sess.run(self.net._layers['end_points']['resnet_v1_101/block2/unit_3/bottleneck_v1/conv1'], feed_dict=feed_dict)
        feat77 = feat7.view().reshape(feat7.shape[0], -1)
        feat8 = sess.run(self.net._layers['end_points']['resnet_v1_101/block3/unit_1/bottleneck_v1/conv1'], feed_dict=feed_dict)
        feat88 = feat8.view().reshape(feat8.shape[0], -1)
        #print(feat11.shape)"""
      else:
        # Compute the graph without summary
        rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, total_loss = \
          self.net.train_step(sess, blobs, train_op)
        #wandb.log({'iteration': iter, 'loss': total_loss})
        #config = wandb.config
        #config.learning_rate = rate
        #config.total_loss = total_loss
        wandb.log({'rpn_loss_cls': rpn_loss_cls, 'rpn_loss_box': rpn_loss_box, 'total_loss': total_loss})
        timer.toc()
        #wandb.tensorflow.log(tf.summary.merge_all())
        #wandb.tensorflow.log(summary)
        #estimator.train(hooks=[wandb.tensorflow.WandbHook(steps_per_log=1000)])
        #ouput1 = self.net.model_summary(sess, blobs, iter, max_iters, train_op)
        """feed_dict = {self.net._image: blobs['data'], self.net._im_info: blobs['im_info'],
                 self.net._gt_boxes: blobs['gt_boxes']}
        feat1 = sess.run(self.net._layers['end_points']['resnet_v1_101/block2/unit_1/bottleneck_v1/conv1'], feed_dict=feed_dict)
        #feat11 = tf.concat([feat11, (feat1.reshape(feat1.shape[0], -1))], 0)
        feat11 = np.concatenate([feat11, feat1.view().reshape(feat1.shape[0], -1)],0)
        feat2 = sess.run(self.net._layers['end_points']['resnet_v1_101/block2/unit_1/bottleneck_v1/conv2'], feed_dict=feed_dict)
        #feat22 = tf.concat([feat22, (feat2.reshape(feat2.shape[0], -1))], 0)
        feat22 = np.concatenate([feat22, feat2.view().reshape(feat2.shape[0], -1)],0)
        feat3 = sess.run(self.net._layers['end_points']['resnet_v1_101/block2/unit_1/bottleneck_v1/conv3'], feed_dict=feed_dict)
        #feat33 = tf.concat([feat33, (feat3.reshape(feat3.shape[0], -1))], 0)
        feat33 = np.concatenate([feat33, feat3.view().reshape(feat3.shape[0], -1)],0)
        feat4 = sess.run(self.net._layers['end_points']['resnet_v1_101/block2/unit_2/bottleneck_v1/conv1'], feed_dict=feed_dict)
        feat44 = np.concatenate([feat44, feat4.view().reshape(feat4.shape[0], -1)],0)
        feat5 = sess.run(self.net._layers['end_points']['resnet_v1_101/block2/unit_2/bottleneck_v1/conv2'], feed_dict=feed_dict)
        feat55 = np.concatenate([feat55, feat5.view().reshape(feat5.shape[0], -1)],0)
        feat6 = sess.run(self.net._layers['end_points']['resnet_v1_101/block2/unit_2/bottleneck_v1/conv3'], feed_dict=feed_dict)
        feat66 = np.concatenate([feat66, feat6.view().reshape(feat6.shape[0], -1)],0)
        feat7 = sess.run(self.net._layers['end_points']['resnet_v1_101/block2/unit_3/bottleneck_v1/conv1'], feed_dict=feed_dict)
        feat77 = np.concatenate([feat77, feat7.view().reshape(feat7.shape[0], -1)],0)
        feat8 = sess.run(self.net._layers['end_points']['resnet_v1_101/block3/unit_1/bottleneck_v1/conv1'], feed_dict=feed_dict)
        feat88 = np.concatenate([feat88, feat8.view().reshape(feat8.shape[0], -1)],0)
        #print(feat11.shape)
      timer.toc()
      #print(feat11.shape)
      #print(feat22.shape)
      #print(feat33.shape)
      ID = {
              "feat11": feat11,
              "feat22": feat22,
              "feat33": feat33,
              "feat44": feat44,
              "feat55": feat55,
              "feat66": feat66,
              "feat77": feat77,
              "feat88": feat88
              }"""
      
      #print("here")
      #print(self.net.model_summary())

      # Display training information
      if iter % (cfg.TRAIN.DISPLAY) == 0:
        print('iter: %d / %d, total loss: %.6f\n >>> rpn_loss_cls: %.6f\n '
              '>>> rpn_loss_box: %.6f\n >>> loss_cls: %.6f\n >>> loss_box: %.6f\n >>> lr: %f' % \
              (iter, max_iters, total_loss, rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, lr.eval()))
        print('speed: {:.3f}s / iter'.format(timer.average_time))

      # Snapshotting
      if iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
        last_snapshot_iter = iter
        ss_path, np_path = self.snapshot(sess, iter)
        np_paths.append(np_path)
        ss_paths.append(ss_path)

        # Remove the old snapshots if there are too many
        if len(np_paths) > cfg.TRAIN.SNAPSHOT_KEPT:
          self.remove_snapshot(np_paths, ss_paths)

      iter += 1

    """ID_all = []
    for key, value in ID.iteritems():
        ID_all.append(self.net.computeID(value, 20, 0.9))
    ID_all = np.array(ID_all)
    print("Final result: {}", format(ID_all[:,0]))
    print("Done.")"""
        
     
    #print(feat11.shape)
    #print(feat22.shape)
    #print(feat33.shape)
    
    
    #print(self.net.computeID(feat11, 20, 0.9))
    #print(self.net.computeID(feat22, 20, 0.9))
    #print(self.net.computeID(feat33, 20, 0.9))
    #print(self.net.computeID(feat44, 20, 0.9))
    #print(self.net.computeID(feat55, 20, 0.9))
    #print(self.net.computeID(feat66, 20, 0.9))
    #print(self.net.computeID(feat77, 20, 0.9))
    #print(self.net.computeID(feat88, 20, 0.9))


    if last_snapshot_iter != iter - 1:
      self.snapshot(sess, iter - 1)

    self.writer.close()
    self.valwriter.close()


def get_training_roidb(imdb, noise):
  """Returns a roidb (Region of Interest database) for use in training."""
  if cfg.TRAIN.USE_FLIPPED:
    print('Appending horizontally-flipped training examples...')
    imdb.append_flipped_images()
    print('done')

  print('Preparing training data...')
  rdl_roidb.prepare_roidb(imdb, noise)
  print('done')

  return imdb.roidb


def filter_roidb(roidb):
  """Remove roidb entries that have no usable RoIs."""

  def is_valid(entry):
    # Valid images have:
    #   (1) At least one foreground RoI OR
    #   (2) At least one background RoI
    #print(entry)
    overlaps = entry['max_overlaps']
    # find boxes with sufficient overlap
    fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # image is only valid if such boxes exist
    valid = len(fg_inds) > 0 or len(bg_inds) > 0
    return valid

  num = len(roidb)
  filtered_roidb = [entry for entry in roidb if is_valid(entry)]
  num_after = len(filtered_roidb)
  print('Filtered {} roidb entries: {} -> {}'.format(num - num_after,
                                                     num, num_after))
  return filtered_roidb


def train_net(network, imdb, roidb, valroidb, output_dir, tb_dir,
              pretrained_model=None,
              max_iters=40000):
  """Train a Faster R-CNN network."""
  roidb = filter_roidb(roidb)
  valroidb = filter_roidb(valroidb)

  tfconfig = tf.ConfigProto(allow_soft_placement=True)
  #tfconfig = tf.ConfigProto()
  tfconfig.gpu_options.allow_growth = True
  

  with tf.Session(config=tfconfig) as sess:
    sw = SolverWrapper(sess, network, imdb, roidb, valroidb, output_dir, tb_dir,
                       pretrained_model=pretrained_model)
    print('Solving...')
    sw.train_model(sess, max_iters)
    #wandb.tensorflow.log(tf.summary.merge_all())
    print('done solving')
