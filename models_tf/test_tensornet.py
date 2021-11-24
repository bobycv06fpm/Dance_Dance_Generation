#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 20:00:06 2019

@author: user
"""



import tensorflow as tf
import tensornets as nets

inputs = tf.placeholder(tf.float32, [None, 224, 224, 3])
model = nets.ResNet50(inputs)

assert isinstance(model, tf.Tensor)
#img = nets.utils.load_img('cat.png')
#assert img.shape == (1, 224, 224, 3)

