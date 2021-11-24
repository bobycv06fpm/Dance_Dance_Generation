#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 16:01:19 2019

@author: aksrustagi
"""
from __future__ import print_function

#import argparse
#from datetime import datetime
#import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

#import sys
import time

#from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from BodyParsing.deeplab_resnet import DeepLabResNetModel #, ImageReader, decode_labels, prepare_label

#import pdb

"""*********************************"""
"""*********************************"""
from stn import spatial_transformer_network as transformer
"""*********************************"""
"""*********************************"""
#import torch
#from torch.autograd import Variable
#import torch.nn.functional as F
#import torchvision.transforms as transforms
#
#import torch.nn as nn
#import torch.utils.data
#import numpy as np
#from AlphaPose.opt import opt
#
#from dataloader import ImageLoader, DetectionLoader, DetectionProcessor, DataWriter, Mscoco
#from AlphaPose.yolo.util import write_results, dynamic_write_results
#from AlphaPose.SPPE.src.main_fast_inference import *
#
#import os
#import sys
#from tqdm import tqdm
#import time
#from AlphaPose.fn import getTime
#
#from AlphaPose.pPose_nms import pose_nms, write_json

import cv2
"""
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
"""
# input video from profi-dancer;) : 
Filename = '/home/aksrustagi/Desktop/Video_Input/Video3/1.mp4'
Cap = cv2.VideoCapture(Filename)

def get_frame(frame_number):
    Cap.set(1,frame_number)
    ret, frame = Cap.read()
#    input_image, real_image=get_heatmap(frame)
    return frame #input_image, real_image

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Sample Frames')
ax1.imshow(cv2.cvtColor(get_frame(100), cv2.COLOR_BGR2RGB))
ax2.imshow(cv2.cvtColor(get_frame(305), cv2.COLOR_BGR2RGB))
fn='./output/res_step3/'+str(1)+'.png'
#plt.show()
fig.savefig(fn)
plt.close() 

tf.reset_default_graph()

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
    
NUM_CLASSES = 7


RESTORE_FROM = '/aibridge/DDG4_ALp/BodyParsing/models/final_model/model.ckpt-19315'


def load(saver, sess, ckpt_path):
    '''Load trained weights.
    
    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    ''' 
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))



image_batch = tf.expand_dims(tf.convert_to_tensor(get_frame(100),dtype=tf.float32), dim=0)# Add one batch dimension.

print(image_batch.shape)
# Create network.
net = DeepLabResNetModel({'data': image_batch}, is_training=False, num_classes=NUM_CLASSES)
print("**********************************")
print("**********************************")
print(net)
print("**********************************")
# Which variables to load.
restore_var = tf.global_variables()

# Predictions.
raw_output = net.layers['fc1_voc12']
raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(image_batch)[1:3,])
raw_output_up = tf.argmax(raw_output_up, dimension=3)
pred = tf.expand_dims(raw_output_up, dim=3)


# Set up TF session and initialize variables. 
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

#with tf.device("/gpu:0"):

init = tf.global_variables_initializer()
sess.run(init)
# Load weights.
loader = tf.train.Saver(var_list=restore_var)
load(loader, sess, RESTORE_FROM)

start_time = time.time()
with tf.device('/device:GPU:2'):

    preds = sess.run([pred])
total_time = time.time() - start_time

print("**********************************")
print("**********************************")
print(np.array(preds).shape)
print(np.unique(np.array(preds)))
print("**********************************")
plt.imshow(np.squeeze(preds))
fn='./output/res_step3/'+str(2)+'.png'
#plt.show()
plt.savefig(fn)
plt.show()
plt.close()
print('It took {} sec.'.format(total_time))

#msk = decode_labels(preds, num_classes=args.num_classes)
#im = Image.fromarray(msk[0])
#img_o = Image.open(jpg_path)
#jpg_path = jpg_path.decode().split('/')[-1].split('.')[0]
#img = np.array(im)*0.9 + np.array(img_o)*0.7
#img[img>255] = 255
#img = Image.fromarray(np.uint8(img))
#img.save(args.save_dir + jpg_path + '.png')
#print('Image processed {}.png'.format(jpg_path))


#print('It took {} sec on each image.'.format(total_time))



  
  
