#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 23:44:56 2019

@author: user
"""

import tensorflow as tf
import layers as ops

# Global_Network
class Generator1:
  def __init__(self, name, is_training, ngf=64, norm="instance",image_size=256):
    self.name = name
    self.reuse = tf.AUTO_REUSE
    self.ngf = ngf
    self.norm = norm
    self.is_training = is_training
    self.image_size = image_size
  
  def __call__(self, input):
    """
    Args:
      input: batch_size x width x height x 3
    Returns:
      output: same size as input
    """
    with tf.variable_scope(self.name):
      # conv layers
      c7s1_64 = ops.c7s1_k(input, self.ngf, is_training=self.is_training, norm=self.norm,
          reuse=self.reuse, name='G1_c7s1_64')                             # (?, w, h, 64)
      d128 = ops.dk(c7s1_64, 2*self.ngf, is_training=self.is_training, norm=self.norm,
          reuse=self.reuse, name='G1_d128')                                 # (?, w/2, h/2, 128)
      d256 = ops.dk(d128, 4*self.ngf, is_training=self.is_training, norm=self.norm,
          reuse=self.reuse, name='G1_d256')                                # (?, w/4, h/4, 256)
      d512 = ops.dk(d256, 8*self.ngf, is_training=self.is_training, norm=self.norm,
          reuse=self.reuse, name='G1_d512')                                 # (?, w/8, h/8, 512)
      d1024 = ops.dk(d512, 16*self.ngf, is_training=self.is_training, norm=self.norm,
          reuse=self.reuse, name='G1_d1024')                                # (?, w/16, h/16, 1024)                                # (?, w/4, h/4, 256)

      # 9 blocks for higher resolution
      res_output = ops.n_res_blocks(d1024, reuse=self.reuse, n=9)      # (?, w/16, h/16, 1024)

      # fractional-strided convolution
      u512 = ops.uk(res_output, 8*self.ngf, is_training=self.is_training, norm=self.norm,
          reuse=self.reuse, name='G1_u512')                                   # (?, w/8, h/8, 512)
      u256 = ops.uk(u512, 4*self.ngf, is_training=self.is_training, norm=self.norm,
          reuse=self.reuse, name='G1_u256')                                   # (?, w/4, h/4, 256)
      u128 = ops.uk(u256, 2*self.ngf, is_training=self.is_training, norm=self.norm,
          reuse=self.reuse, name='G1_u128')                                   # (?, w/2, h/2, 128)
      u64 = ops.uk(u128, self.ngf, is_training=self.is_training, norm=self.norm,
          reuse=self.reuse, name='G1_u64')                                    # (?, w, h, 64)
      
      print("G1_lastfeature_shape",u64.shape)
      # conv layer
      # Note: the paper said that ReLU and _norm were used
      # but actually tanh was used and no _norm here
      output = ops.c7s1_k(u64, 3, norm=None,
          activation='tanh', reuse=self.reuse, name='G1_output')           # (?, w, h, 3)
    # set reuse=True for next call
    self.reuse = True
    self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    return output, u64
    
  




###Enhancer_Network

class Generator2:
    
  def __init__(self, name="G2", is_training=False, ngf=32, norm="instance", image_size=256):
      
    self.name = name
    self.reuse = tf.AUTO_REUSE
    self.ngf = ngf
    self.norm = norm
    self.is_training = is_training
    self.image_size = image_size
    self.name_G1 = "G1"
  def __call__(self, input):
    """
    Args:
      input: batch_size x width x height x 3
    Returns:
      output: same size as input
    """
    with tf.variable_scope(self.name):
        with tf.variable_scope("G1"):
            input_G1 = tf.layers.average_pooling2d(input, pool_size=3, strides=2, padding='same')
            G1 = Generator1('Gen1', self.is_training, ngf=2*self.ngf, norm=self.norm)
            output_G1 , last_feat = G1(input_G1)
            self.variables1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+'/G1/Gen1')


        # conv layers
        c7s1_32 = ops.c7s1_k(input, self.ngf, is_training=self.is_training, norm=self.norm,
                      reuse=self.reuse, name='G2_c7s1_32')                             # (?, w, h, 64)
        d64 = ops.dk(c7s1_32, 2*self.ngf, is_training=self.is_training, norm=self.norm,
                      reuse=self.reuse, name='G2_d64')                                 # (?, w/2, h/2, 64)
    
        # element-wise sum of two feature maps: the output feature map of downsample_G2
        # and the last feature map of the back-end of the global generator network
        resnet_input = last_feat + d64
        # 3 blocks for higher resolution
        res_output = ops.n_res_blocks(resnet_input, reuse=self.reuse, n=3)      # (?, w/2, h/2, 64)
            
        # fractional-strided convolution
        u32 = ops.uk(res_output, self.ngf, is_training=self.is_training, norm=self.norm,
                      reuse=self.reuse, name='G2_u32')                                   # (?, w, h, 32)
        # conv layer
        # Note: the paper said that ReLU and _norm were used
        # but actually tanh was used and no _norm here
        output = ops.c7s1_k(u32, 3, norm=None,
                      activation='tanh', reuse=self.reuse, name='G2_output')           # (?, w, h, 3)
        # set reuse=True for next call
        self.reuse = True
        self.variables2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        
        return output


