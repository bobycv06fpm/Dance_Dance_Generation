#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 16:01:19 2019

@author: aksrustagi
"""
from __future__ import print_function

#import argparse
#from datetime import datetime
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152

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
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms

import torch.nn as nn
import torch.utils.data
import numpy as np
from AlphaPose.opt import opt

from dataloader import ImageLoader, DetectionLoader, DetectionProcessor, DataWriter, Mscoco
from AlphaPose.yolo.util import write_results, dynamic_write_results
from AlphaPose.SPPE.src.main_fast_inference import *

import os
import sys
from tqdm import tqdm
from AlphaPose.fn import getTime

from AlphaPose.pPose_nms import pose_nms, write_json

import cv2
"""
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
"""
# input video from profi-dancer;) : 
Filename = '/home/aksrustagi/Desktop/Video_Input/Video3/1.mp4'
Cap = cv2.VideoCapture(Filename)
#torchCuda=2
def get_frame(frame_number):
    Cap.set(1,frame_number)
    ret, frame = Cap.read()
#    input_image, real_image=get_heatmap(frame)
    return frame #input_image, real_image
plt.figure()
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Sample Frames')
ax1.imshow(cv2.cvtColor(get_frame(100), cv2.COLOR_BGR2RGB))
ax2.imshow(cv2.cvtColor(get_frame(305), cv2.COLOR_BGR2RGB))
fn='./output/res_step3/'+str(1)+'.png'
#plt.show()
fig.savefig(fn)
plt.close() 


"""
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
"""
print('__Python VERSION:', sys.version)
print('__pyTorch VERSION:', torch.__version__)
#print('__CUDA VERSION')
#from subprocess import call
# call(["nvcc", "--version"]) does not work
print('__CUDNN VERSION:', torch.backends.cudnn.version())
print('__Number CUDA Devices:', torch.cuda.device_count())
print('__Devices')
print('Active CUDA Device: GPU', torch.cuda.current_device())
print ('Available devices ', torch.cuda.device_count())
print ('Current cuda device ', torch.cuda.current_device())


# Load input images
Frame1=get_frame(100)
img, orig_img, im_dim_list= ImageLoader.cnv_img(Frame1)

# Load detection loader
print('Loading YOLO model..')
orig_img,boxes,scores,inps,pt1,pt2 = DetectionLoader().load(img, orig_img, im_dim_list)
orig_img,boxes,scores,inps,pt1,pt2 = DetectionProcessor.process(orig_img,boxes,scores,inps,pt1,pt2)


# Load pose model
pose_dataset = Mscoco()
if True: #args.fast_inference:
    pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset)
else:
    pose_model = InferenNet(4 * 1 + 1, pose_dataset)
pose_model.cuda()
pose_model.eval()


batchSize = 80 #args.posebatch

start_time = getTime()
with torch.no_grad():
#    (inps, orig_img, im_name, boxes, scores, pt1, pt2) = det_processor.read()
    # Pose Estimation            
    hm = []
    j=0 
    inps_j = inps[j*batchSize:min((j +  1)*batchSize, 1)].cuda()
    hm_j = pose_model(inps_j)
    hm.append(hm_j)
    hm = torch.cat(hm)
    hm = hm.cpu()
    img,joints = DataWriter.update(boxes, scores, hm, pt1, pt2, orig_img)

plt.figure()
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
fn='./output/res_step3/'+str(2)+'.png'
#plt.show()
plt.savefig(fn)
#plt.show()
plt.close()
  
"""
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
"""     
