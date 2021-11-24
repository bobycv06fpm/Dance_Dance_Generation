import tensorflow as tf
import layers as ops

class Discriminator:
  def __init__(self, name, is_training=False, norm="instance", use_sigmoid=False):
    self.name = name
    self.is_training = is_training
    self.norm = norm
    self.reuse = tf.AUTO_REUSE
    self.use_sigmoid = use_sigmoid

  def __call__(self, input):
    """
    Args:
      input: batch_size x image_size x image_size x 3
    Returns:
      output: 4D tensor batch_size x out_size x out_size x 1 (default 1x5x5x1)
              filled with 0.9 if real, 0.0 if fake
    """

    with tf.variable_scope(self.name):
      # convolution layers
      C64 = ops.Ck(input, 64, reuse=self.reuse, norm=None,
          is_training=self.is_training, name='C64')                     # (?, w/2, h/2, 64)
      C128 = ops.Ck(C64, 128, reuse=self.reuse, norm=self.norm,
          is_training=self.is_training, name='C128')                    # (?, w/4, h/4, 128)

      C256 = ops.Ck(C128, 256, reuse=self.reuse, norm=self.norm,
          is_training=self.is_training, name='C256')                    # (?, w/8, h/8, 256)
      C512 = ops.Ck(C256, 512,reuse=self.reuse, norm=self.norm,
          is_training=self.is_training, name='C512')                    # (?, w/16, h/16, 512)

      # apply a convolution to produce a 1 dimensional output (1 channel?)
      # use_sigmoid = False if use_lsgan = True
      output = ops.last_conv(C512, reuse=self.reuse,
          use_sigmoid=self.use_sigmoid, name='output') 
#    print("OKKKKKKKKKKKKKKKKKKKKKKKK_______________________________",output.shape) # (?, w/16, h/16, 1)
      
    self.reuse = True
    self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    return [output,C64, C128, C256, C512]


class multi_scale_Discriminator:
  def __init__(self, name="D", is_training=False, norm="instance", use_sigmoid=False):
    self.name = name
    self.is_training = is_training
    self.norm = norm
    self.reuse = tf.AUTO_REUSE
    self.use_sigmoid = use_sigmoid

  def __call__(self, input):
      
    with tf.variable_scope(self.name):
            
        D2_input = tf.layers.average_pooling2d(input, pool_size=3, strides=2, padding='same')
        D3_input = tf.layers.average_pooling2d(D2_input, pool_size=3, strides=2, padding='same')
        D1 = Discriminator("D1", self.is_training, self.norm, self.use_sigmoid)
        D2 = Discriminator("D2", self.is_training, self.norm, self.use_sigmoid)
        D3 = Discriminator("D3", self.is_training, self.norm, self.use_sigmoid)
        output_D1 = D1(input)
        output_D2 = D2(D2_input)
        output_D3 = D3(D3_input)

    self.reuse = True
    self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
    return output_D1, output_D2, output_D3




 


