from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib.pyplot as plt

import numpy as np
import os

import cv2
import torch
import tensorflow as tf

from pytorch_OpenPose.network.rtpose_vgg import get_model
#from network.post import decode_pose

from pytorch_OpenPose.evaluate.coco_eval import get_multiplier, get_outputs #, handle_paf_and_heat




# input video from profi-dancer;) : 
Filename = './Video_Input/Video8/1.mp4'




weight_name = './pytorch_OpenPose/network/weight/pose_model.pth'
model = get_model('vgg19')
model.load_state_dict(torch.load(weight_name))
model = torch.nn.DataParallel(model).cuda()

model.float()
model.eval()
print('Model loaded . . .')


def get_heatmap(oriImg):
    real_image=oriImg
    multiplier = get_multiplier(oriImg)    
    with torch.no_grad():
        orig_paf, input_image = get_outputs(
            multiplier, oriImg, model,  'rtpose')
        
    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return input_image[...,:-1], real_image


Cap = cv2.VideoCapture(Filename)

BUFFER_SIZE = 1000

IMG_WIDTH = 256
IMG_HEIGHT = 256

def load(image_file):
    Cap.set(1,image_file)
    ret, frame = Cap.read()
    input_image, real_image=get_heatmap(frame)
    return input_image, real_image


def resize(input_image, real_image, height, width):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return input_image, real_image

# normalizing the images to [-1, 1]

def normalize(input_image, real_image):
  r_mean=tf.math.reduce_mean(input_image )
  r_var =tf.math.reduce_mean(tf.square(input_image-r_mean))
  input_image = (input_image - r_mean) / r_var
  real_image = (real_image / 127.5) - 1

  return input_image, real_image

def load_image_test(image_file):
  input_image, real_image = load(image_file)
  input_image, real_image = resize(input_image, real_image,
                                   IMG_HEIGHT, IMG_WIDTH)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image


HeatMaps=np.zeros((BUFFER_SIZE,IMG_HEIGHT, IMG_WIDTH,18))
TargetImages=np.zeros((BUFFER_SIZE,IMG_HEIGHT, IMG_WIDTH,3))
idx=0
plt.figure()
for frm in range(100,1100):
    input_image, real_image = load_image_test(frm)
    HeatMaps[idx,...]=input_image
    TargetImages[idx,...]=real_image
    idx+=1
    print('Heatmaps for Frame =>',frm)

''' %%%%%%%%%%%%%%%
    Creating Test Graph for GAN 
    %%%%%%%%%%%%%%% '''
OUTPUT_CHANNELS = 3    
def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result


def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result


def Generator():
  down_stack = [
    downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
    downsample(128, 4), # (bs, 64, 64, 128)
    downsample(256, 4), # (bs, 32, 32, 256)
    downsample(512, 4), # (bs, 16, 16, 512)
    downsample(512, 4), # (bs, 8, 8, 512)
    downsample(512, 4), # (bs, 4, 4, 512)
    downsample(512, 4), # (bs, 2, 2, 512)
    downsample(512, 4), # (bs, 1, 1, 512)
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
    upsample(512, 4), # (bs, 16, 16, 1024)
    upsample(256, 4), # (bs, 32, 32, 512)
    upsample(128, 4), # (bs, 64, 64, 256)
    upsample(64, 4), # (bs, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh') # (bs, 256, 256, 3)

  concat = tf.keras.layers.Concatenate()

  inputs = tf.keras.layers.Input(shape=[None,None,18])
  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = concat([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

generator = Generator()

checkpoint_dir = './training_checkpoints'
checkpoint = tf.train.Checkpoint(generator=generator)

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

def generate_images(model, test_input, tar):
  # the training=True is intentional here since
  # we want the batch statistics while running the model
  # on the test dataset. If we use training=False, we will get
  # the accumulated statistics learned from the training dataset
  # (which we don't want)
  prediction = model(test_input, training=True)
  plt.figure(figsize=(15,15))

#  display_list = [np.sum(test_input,axis=2), tar[0], prediction[0]]
  title = ['Input Image', 'Ground Truth', 'Predicted Image']
  i=0
  plt.subplot(1, 3, i+1)
  plt.title(title[i])
  plt.imshow(np.sum(test_input[0],axis=2) * 0.5 + 0.5)
  plt.axis('off')
  i=1
  plt.subplot(1, 3, i+1)
  plt.title(title[i])
  plt.imshow(cv2.cvtColor(np.float32(tar[0] * 0.5 + 0.5),cv2.COLOR_BGR2RGB))
  plt.axis('off')
  i=2
  plt.subplot(1, 3, i+1)
  plt.title(title[i])
  Pred=cv2.cvtColor(np.float32(prediction[0] * 0.5 + 0.5),cv2.COLOR_BGR2RGB)
  plt.imshow(Pred)
  plt.axis('off')
  fn='./OP_Dance_ImShow/'+str(np.random.choice(10))+'.png'
  plt.savefig(fn)
  plt.close()
  return np.float32(prediction[0] * 0.5 + 0.5)

''' %%%%%%%%%%%%%%%
    Testing Model 
    %%%%%%%%%%%%%%%'''
testFilename = './Video_Output/test.mp4'

Cap.set(1,1)
ret, tar = Cap.read()

height, width, layers = tar.shape
size = (width,height)
fps=Cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter(testFilename,fourcc, fps, size)

idx=0
for frm in range(100,1100):
    inp = HeatMaps[idx,...]
    tar = TargetImages[idx,...]
    inp = tf.cast(inp, tf.float32)
    tar = tf.cast(tar, tf.float32)    
    Pred=generate_images(generator, inp[tf.newaxis,...], tar[tf.newaxis,...])
    NewFrame=255*(cv2.resize(Pred, dsize=(width,height), interpolation=cv2.INTER_CUBIC))
    tmp_img = NewFrame.astype(np.uint8)
    out.write(tmp_img)
    print('Frame =>',frm)
    idx+=1

out.release()