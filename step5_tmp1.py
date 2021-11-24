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

#import sys
import time

"""*********************************"""
"""*********************************"""
from stn import spatial_transformer_network as transformer
"""*********************************"""
"""*********************************"""
import tensorflow as tf
#tf.logging.set_verbosity(tf.logging.WARN)
#from PIL import Image
import matplotlib.pyplot as plt
#import tensorflow as tf
import numpy as np

from BodyParsing.deeplab_resnet import DeepLabResNetModel #, ImageReader, decode_labels, prepare_label

#import pdb

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
Filename = '/home/aksrustagi/Desktop/Video_Input/Video7/1.mp4'
Cap = cv2.VideoCapture(Filename)
torchCuda=0
def get_frame(frame_number):
    Cap.set(1,frame_number)
    ret, frame = Cap.read()
#    input_image, real_image=get_heatmap(frame)
    return frame #input_image, real_image



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
# Load detection loader
print('Loading YOLO model..')
Dloader=DetectionLoader()
# Load pose model
pose_dataset = Mscoco()
if True: #args.fast_inference:
    pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset)
else:
    pose_model = InferenNet(4 * 1 + 1, pose_dataset)
pose_model.cuda(torchCuda)
pose_model.eval()

def get_joints(Frame1):
    # Load input images
    #Frame1=get_frame(100)
    img, orig_img, im_dim_list= ImageLoader.cnv_img(Frame1)
    orig_img,boxes,scores,inps,pt1,pt2 = Dloader.load(img, orig_img, im_dim_list)
    orig_img,boxes,scores,inps,pt1,pt2 = DetectionProcessor.process(orig_img,boxes,scores,inps,pt1,pt2)
    batchSize = 80 #args.posebatch
    #start_time = getTime()
    with torch.no_grad():
    #    (inps, orig_img, im_name, boxes, scores, pt1, pt2) = det_processor.read()
        # Pose Estimation            
        hm = []
        j=0 
        inps_j = inps[j*batchSize:min((j +  1)*batchSize, 1)].cuda(torchCuda)
        hm_j = pose_model(inps_j)
        hm.append(hm_j)
        hm = torch.cat(hm)
        hm = hm.cpu()
        img,joints = DataWriter.update(boxes, scores, hm, pt1, pt2, orig_img)
    return img,joints


  
"""
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
"""     
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


with tf.device('/device:GPU:2'):
    
    image_batch = tf.placeholder(tf.float32, shape=[None,get_frame(2).shape[0],get_frame(2).shape[1],3])#tf.expand_dims(tf.convert_to_tensor(get_frame(100),dtype=tf.float32), dim=0)# Add one batch dimension.
    print(image_batch.shape)
    # Create network.
    net = DeepLabResNetModel({'data': image_batch}, is_training=False, num_classes=NUM_CLASSES)
    # Which variables to load.
    restore_var = tf.global_variables()
    
    # Predictions.
    raw_output = net.layers['fc1_voc12']
    raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(image_batch)[1:3,])
    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    pred = tf.expand_dims(raw_output_up, dim=3)
    
#    H, W, C = get_frame(1).shape
#    
#    stn_img = tf.placeholder(tf.float32, [None, H, W, C])
#    stn_theta = tf.placeholder(tf.float32, [None, 6])
#    stn_res= transformer(stn_img, stn_theta) 
    
    # Set up TF session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    
    #with tf.device("/gpu:0"):
    
#    init = tf.global_variables_initializer()
#    sess.run(init)
# Load weights.
loader = tf.train.Saver(var_list=restore_var)
load(loader, sess, RESTORE_FROM)

def get_bodymap(image_b):
    with tf.device('/device:GPU:2'):
#        image_b = get_frame(200)
        image_b=image_b[np.newaxis,...]
#        start_time = time.time()
        preds = sess.run([pred],feed_dict={image_batch:image_b})
    return np.squeeze(preds)
#        total_time = time.time() - start_time

print("**********************************")
print("**********************************")
preds=get_bodymap(get_frame(100))
print(np.array(preds).shape)
print(np.unique(np.array(preds)))
print("**********************************")

#print('It took {} sec.'.format(total_time))

def old2new_joint(joint):
    new_j=np.zeros((15,2))
    #head
    new_j[0,:]=np.mean(joint[0:5,:],axis=0)
    #neck
    new_j[1,:]=np.mean(joint[5:7,:],axis=0)
    #torso
    new_j[2,:]=np.mean(joint[11:13,:], axis=0)
    #left shoulder
    new_j[3,:]=joint[5,:]
    #right shoulder
    new_j[4,:]=joint[6,:]
    #left elbow
    new_j[5,:]=joint[7,:]
    #right elbow
    new_j[6,:]=joint[8,:]    
    #left wrist
    new_j[7,:]=joint[9,:]
    #right wrist
    new_j[8,:]=joint[10,:] 
    #left hip
    new_j[9,:]=joint[11,:]
    #right hip
    new_j[10,:]=joint[12,:] 
    #left knee
    new_j[11,:]=joint[13,:]
    #right knee
    new_j[12,:]=joint[14,:]  
    #left ankle
    new_j[13,:]=joint[15,:]
    #right ankle
    new_j[14,:]=joint[16,:]    
    return new_j

def old2new_mask(mask,joints,skeletonLines):
    def seperate_lr(t_msk,p1,p2):
        t_msk=t_msk.astype(np.float32)
        p1=[p1[1],p1[0]]
        p2=[p2[1],p2[0]]
        res_msk=np.zeros_like(t_msk)
        idx=np.transpose(np.nonzero(t_msk))
        dist=np.zeros((2,len(idx)))
        idx=np.array(idx)
        dist[0,:]=np.linalg.norm(idx-p1,axis=-1)
        dist[1,:]=np.linalg.norm(idx-p2,axis=-1)
#        print(idx.shape)
#        print(dist.shape)
        lbl=np.argmin(dist,axis=0)
#        print(lbl.shape)
        for i in range(idx.shape[0]):
            res_msk[idx[i,0],idx[i,1]]=(lbl[i]+1)

        return res_msk    
    #old    #head 1    #body 2    #arm  3    #wrist4    #thigh5    #leg  6
    #convert to
    #new    #head 1    #body 2    #L arm 3    #R arm 4    #L wrist 5
    #R wirst 6    #L thigh 7    #R thigh 8    #L leg 9    #R leg 10
    new_m=np.zeros_like(mask)
    new_m[mask==1]=1
    new_m[mask==2]=2
    #new arms
    p1=np.mean(joints[skeletonLines[2],:],axis=0)
    p2=np.mean(joints[skeletonLines[3],:],axis=0)
    tmsk=seperate_lr(mask==3,p1,p2)
    new_m[tmsk==1]=3
    new_m[tmsk==2]=4
    #new wrist
    p1=np.mean(joints[skeletonLines[4],:],axis=0)
    p2=np.mean(joints[skeletonLines[5],:],axis=0)
    tmsk=seperate_lr(mask==4,p1,p2)
    new_m[tmsk==1]=5
    new_m[tmsk==2]=6  
    #new thigh
    p1=np.mean(joints[skeletonLines[6],:],axis=0)
    p2=np.mean(joints[skeletonLines[7],:],axis=0)
    tmsk=seperate_lr(mask==5,p1,p2)
    new_m[tmsk==1]=7
    new_m[tmsk==2]=8  
    #new leg
    p1=np.mean(joints[skeletonLines[8],:],axis=0)
    p2=np.mean(joints[skeletonLines[9],:],axis=0)
    tmsk=seperate_lr(mask==6,p1,p2)
    new_m[tmsk==1]=9
    new_m[tmsk==2]=10    
    return new_m

        
        

def draw_pose(joints,skeletonLines):
    img=np.zeros_like(get_frame(1),dtype=np.uint8)+255
    pose = joints
    for i in range(len(skeletonLines)):
        c = [np.random.choice(256),np.random.choice(256),np.random.choice(256)]
        a = skeletonLines[i][0]
        b = skeletonLines[i][1]
        cv2.line(img, (int(pose[a,0]), int(pose[a,1])), (int(pose[b,0]), int(pose[b,1])), c, 3)
    return img


def get_M_matirx(origin_pose, target_pose,skeletonLines): 
#    origin_pose=np.fliplr(origin_pose)
#    target_pose=np.fliplr(target_pose)
    M=np.zeros((10,2,3))
    TM=np.zeros((10,4))
    for label in range(0, 10):
        origin_size = np.array([get_frame(1).shape[1], get_frame(1).shape[0]])
        a = skeletonLines[label][0]
        b = skeletonLines[label][1]
        origin_pose_part_a = np.array([origin_pose[a ,0], origin_pose[a, 1]])
        origin_pose_part_b = np.array([origin_pose[b ,0], origin_pose[b, 1]])
        origin_pose_part_tensor = origin_pose_part_b - origin_pose_part_a
        target_pose_part_a = np.array([target_pose[a ,0], target_pose[a ,1]])
        target_pose_part_b = np.array([target_pose[b ,0], target_pose[b ,1]])
        target_pose_part_tensor = target_pose_part_b - target_pose_part_a
        origin_pose_part_length = np.sqrt(np.sum(np.square(origin_pose_part_tensor)))
        target_pose_part_length = np.sqrt(np.sum(np.square(target_pose_part_tensor)))
            # scaling ratio
        scale_factor = target_pose_part_length / origin_pose_part_length
            # if scale_factor == 0:
            #     continue
            # rotating angle
     
        theta = -(np.arctan2(target_pose_part_tensor[1], target_pose_part_tensor[0]) - np.arctan2(
                origin_pose_part_tensor[1], origin_pose_part_tensor[0])) * 180 / np.pi
                
        origin_pose_part_center = (origin_pose_part_a + origin_pose_part_b) / 2
        origin_center = origin_size / 2
        tx1 = origin_center[0] - int(origin_pose_part_center[0])
        ty1 = origin_center[1] - int(origin_pose_part_center[1])
#        tm = np.float32([[1, 0, tx], [0, 1, ty]])
        target_pose_part_center = (target_pose_part_a + target_pose_part_b) / 2       
        tx2 = origin_center[0] - int(target_pose_part_center[0])
        ty2 = origin_center[1] - int(target_pose_part_center[1])
        
        rm = cv2.getRotationMatrix2D((origin_size[0]/2, origin_size[1]/2),theta, scale_factor)
#        t=transformer(origin_body_part,rm)
        M[label,...]=rm
        TM[label,...]=[tx1,ty1,tx2,ty2]
    return M,TM

###################################################################################################
def body_transfer(img, mask,M,TM):
#    img = img.astype(np.float32)
    H, W, C = img.shape
    img_new=np.zeros_like(img).astype('float32')
#    img_batch=np.zeros((10,H,W,C))
#    M_batch=np.zeros((10,6))
    for label in range(0, 10):
        part_mask= (mask==(label+1)).astype(np.float32)        
        part_img = img * part_mask[...,None]
        [tx1,ty1,tx2,ty2]=TM[label,...]
        tm1 = np.float32([[1, 0, tx1], [0, 1, ty1]])
        part_img = cv2.warpAffine(part_img, tm1,(W, H))
        part_img = cv2.warpAffine(part_img,M[label,...],(W,H))
        tm2 = np.float32([[1, 0, -tx2], [0, 1, -ty2]])
        part_img = cv2.warpAffine(part_img, tm2,(W, H))        
#        part_img = part_img[np.newaxis,...]
#        img_batch[label,...]=part_img
#        part_M = M[label-1,...].flatten()
#        part_M = part_M[np.newaxis,...]        
#        M_batch[label,...]=part_M
        img_new =  img_new + part_img
#    print(img_batch.shape)  
#    print(M_batch.shape)
#    part_img_new = sess.run(stn_res,feed_dict={stn_img:img_batch,stn_theta:M_batch})
#    
#    for label in range(0, 10):
#        img_new =img_new + part_img_new[label,...]

    return img_new.astype('uint8')

"""
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
Main Code
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
""" 
main_skeleton_lines = [
    [0,1], #head
    [1,2], #body
    [3,5], #L arm
    [4,6], #R arm
    [5,7], #L wrist
    [6,8], #R wirst
    [9,11], #L thigh
    [10,12], #R thigh
    [11,13], #L leg
    [12,14]] #R leg


Frame_in = get_frame(100)
Frame_ref= get_frame(129)
H_in=get_bodymap(Frame_in)
H_ref=get_bodymap(Frame_ref)
timg_in,Pose_in=get_joints(Frame_in)
timg_ref,Pose_ref=get_joints(Frame_ref)
#print(Pose_in)
Old_pose_in=Pose_in['result']
Old_pose_in=np.array(Old_pose_in[0]['keypoints'])
Old_pose_ref=Pose_ref['result']
Old_pose_ref=np.array(Old_pose_ref[0]['keypoints'])
New_pose_ref = old2new_joint(Old_pose_ref)
New_pose_in = old2new_joint(Old_pose_in)


plt.figure()
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Sample Frames')
ax1.imshow(cv2.cvtColor(Frame_in, cv2.COLOR_BGR2RGB))
ax2.imshow(cv2.cvtColor(Frame_ref, cv2.COLOR_BGR2RGB))
fn='./output/res_step3/'+str(1)+'.png'
fig.savefig(fn)
plt.close() 


plt.figure()
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('AlphaPose')
ax1.imshow(cv2.cvtColor(timg_in, cv2.COLOR_BGR2RGB))
ax2.imshow(cv2.cvtColor(timg_ref, cv2.COLOR_BGR2RGB))
fn='./output/res_step3/'+str(2)+'.png'
plt.savefig(fn)
plt.close()


plt.figure()
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('BodyPart6')
ax1.imshow(H_in)
ax2.imshow(H_ref)
fn='./output/res_step3/'+str(3)+'.png'
plt.savefig(fn)
plt.close()

H_in=old2new_mask(H_in,New_pose_in,main_skeleton_lines)
H_in=old2new_mask(H_ref,New_pose_ref,main_skeleton_lines)
plt.figure()
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('BodyPart10')
ax1.imshow(H_in)
ax2.imshow(H_ref)
fn='./output/res_step3/'+str(4)+'.png'
plt.savefig(fn)
plt.close()


dpose_in=draw_pose(New_pose_in,main_skeleton_lines)
dpose_ref=draw_pose(New_pose_ref,main_skeleton_lines)
plt.figure()
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Skeleton')
ax1.imshow(dpose_in)
ax2.imshow(dpose_ref)
fn='./output/res_step3/'+str(5)+'.png'
plt.savefig(fn)
plt.close()


M,TM=get_M_matirx(New_pose_in,New_pose_ref,main_skeleton_lines)
img_transfer=body_transfer(Frame_in,H_in,M,TM)
img_transfer=cv2.cvtColor(img_transfer, cv2.COLOR_BGR2RGB)
plt.figure()
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Results')
ax1.imshow(img_transfer)
tm=(H_ref>0).astype(np.uint8)
ax2.imshow(cv2.cvtColor(Frame_ref, cv2.COLOR_BGR2RGB)*tm[...,None])
fn='./output/res_step3/'+str(6)+'.png'
plt.savefig(fn)
plt.close()

