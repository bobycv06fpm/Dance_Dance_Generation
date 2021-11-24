
from discriminator import multi_scale_Discriminator as Discriminator
from generator import Generator2 as Generator
from generator import Generator1 as Generator1
import layers as ops
import tensorflow as tf
from reader1 import Reader
import utils
#import tensornets as nets
#import skimage
import vgg19
REAL_LABEL = 0.9

class Pix2Pix_model:
  def __init__(self,
               X_train_file='/home/jsh/Downloads/facades/data.tfrecords',
               batch_size=1,
               image_size=256,
               use_lsgan=True,
               norm='instance',
               learning_rate=2e-4,
               beta1=0.5,
               ngf=32
              ):
    """
    Args:
      X_train_file: string, X tfrecords file for training
      batch_size: integer, batch size
      image_size: integer, image size
      use_lsgan: boolean
      norm: 'instance' or 'batch'
      learning_rate: float, initial learning rate for Adam
      beta1: float, momentum term of Adam
      ngf: number of gen filters in first conv layer
    """
    self.use_lsgan = use_lsgan
    use_sigmoid = not use_lsgan
    self.batch_size = batch_size
    self.image_size = image_size
    self.learning_rate = learning_rate
    self.beta1 = beta1
    self.X_train_file = X_train_file

    self.is_training = tf.placeholder_with_default(True, shape=[], name='is_training')
    self.G = Generator('G2', self.is_training, ngf=32, norm=norm, image_size=image_size)
    self.D = Discriminator('D', self.is_training, norm=norm, use_sigmoid=use_sigmoid)
#    self.G1 = Generator1('G1', self.is_training, ngf=32, norm=norm, image_size=image_size)
    self.real_image = tf.placeholder(tf.float32,
        shape=[batch_size, image_size, image_size, 3])
    self.label = tf.placeholder(tf.float32,
        shape=[batch_size, image_size, image_size, 3])


  def model(self):
    fake_image = self.G(self.label)
    G_loss = self.generator_loss(self.D, self.real_image, fake_image) ###???????
    D_loss = self.discriminator_loss(self.D, self.real_image, fake_image)###?????
    vgg_loss = self.VGG19_loss(self.real_image, fake_image)
    feat_loss = self.feature_matching_loss(self.D, self.real_image, fake_image)
    D1_T,D2_T,D3_T =self.D(self.real_image)
    D1_f, D2_f, D3_f = self.D(self.G(self.label))
    Gen_loss= tf.zeros(self.batch_size, tf.float32)
    Gen_loss = tf.reduce_mean(G_loss + 10*vgg_loss + feat_loss)
    # summary
    tf.summary.histogram('D1/true',D1_T[0])
    tf.summary.histogram('D2/true', D2_T[0])
    tf.summary.histogram('D3/true', D3_T[0])
    tf.summary.histogram('D1/fake', D1_f[0])
    tf.summary.histogram('D1/fake', D2_f[0])
    tf.summary.histogram('D1/fake', D3_f[0])
    tf.summary.scalar('loss/G', Gen_loss)
    tf.summary.scalar('loss/D', D_loss)
    tf.summary.scalar('loss/G1', G_loss)
    tf.summary.scalar('loss/vgg', tf.reduce_mean(vgg_loss))
    tf.summary.scalar('loss/feature_matching', tf.reduce_mean(feat_loss))
    tf.summary.image('X/generated', utils.batch_convert2int(fake_image))
    tf.summary.image('X/real', self.real_image)
    tf.summary.image('X/label',self.label)
    return vgg_loss, D_loss, fake_image

  def optimize(self, G_loss, D_loss):
      def make_optimizer(loss, variables, name='Adam'):
          """ Adam optimizer with learning rate 0.0002 for the first 100k steps (~100 epochs)
              and a linearly decaying rate that goes to zero over the next 100k steps
          """
          global_step = tf.Variable(0, trainable=False)
          starter_learning_rate = self.learning_rate
          end_learning_rate = 0.0
          start_decay_step = 100000
          decay_steps = 100000
          beta1 = self.beta1
          learning_rate = (
              tf.where(
                      tf.greater_equal(global_step, start_decay_step),
                      tf.train.polynomial_decay(starter_learning_rate, global_step-start_decay_step,
                                                decay_steps, end_learning_rate,
                                                power=1.0),
                      starter_learning_rate
              )

          )
          tf.summary.scalar('learning_rate/{}'.format(name), learning_rate)

          learning_step = (
              tf.train.AdamOptimizer(learning_rate, beta1=beta1, name=name)
                      .minimize(loss, global_step=global_step, var_list=variables)
          )
          return learning_step

      # D_vars = tf.trainable_variables(scope="D")
      # G1_vars = tf.trainable_variables(scope="G1")
      # G2_vars = tf.trainable_variables(scope="G2")
      G1_optimizer = make_optimizer(G_loss, self.G.variables1, name='Adam_G1')
      G2_optimizer = make_optimizer(G_loss,self.G.variables2, name='Adam_G2')
      D_optimizer = make_optimizer(D_loss, self.D.variables, name='Adam_D')

      with tf.control_dependencies([G1_optimizer,G2_optimizer, D_optimizer]):
#          return tf.no_op(name='optimizers')
           return G1_optimizer, G2_optimizer, D_optimizer



  def discriminator_loss(self,Discriminator, real_image, fake_image, use_lsgan=True):
    """ Note: default: Discriminator(real_image).shape == (batch_size,5,5,1),
                       fake_buffer_size=50, batch_size=1
    Args:
      Discriminator: discriminator object
      real_image: 4D tensor (batch_size, image_size, image_size, 3)
      fake_image: 4D tensor (batch_size, image_size, image_size, 3)
    Returns:
      loss: scalar
"""

    D1_real, D2_real, D3_real = Discriminator(real_image)
    D1_fake, D2_fake, D3_fake = Discriminator(fake_image)

    # relativistic average LSGAN
    D1_fake_deriv = tf.gradients(Discriminator(fake_image)[0],fake_image)
    D2_fake_deriv = tf.gradients(Discriminator(fake_image)[1],fake_image)
    D3_fake_deriv = tf.gradients(Discriminator(fake_image)[2],fake_image)
    error_p1_D1 = tf.reduce_mean(tf.squared_difference(tf.math.subtract(D1_real[0], tf.reduce_mean(D1_fake[0],axis=0)), 1))
    error_p1_D2 = tf.reduce_mean(tf.squared_difference(tf.math.subtract(D2_real[0], tf.reduce_mean(D2_fake[0],axis=0)), 1))
    error_p1_D3 = tf.reduce_mean(tf.squared_difference(tf.math.subtract(D3_real[0], tf.reduce_mean(D3_fake[0],axis=0)), 1))
          
    error_p2_D1 = tf.reduce_mean(tf.squared_difference(D1_fake[0], tf.reduce_mean(D1_real[0],axis=0)))
    error_p2_D2 = tf.reduce_mean(tf.squared_difference(D2_fake[0], tf.reduce_mean(D2_real[0],axis=0)))
    error_p2_D3 = tf.reduce_mean(tf.squared_difference(D3_fake[0], tf.reduce_mean(D3_real[0],axis=0)))
          
    error_p3_D1 = tf.reduce_mean(tf.squared_difference(tf.norm(D1_fake_deriv), 1))
    error_p3_D2 = tf.reduce_mean(tf.squared_difference(tf.norm(D2_fake_deriv), 1))
    error_p3_D3 = tf.reduce_mean(tf.squared_difference(tf.norm(D3_fake_deriv), 1))
    

    loss =tf.reduce_mean((error_p1_D1 + error_p1_D2 + error_p1_D3 + error_p2_D1 +  error_p2_D2 +  error_p2_D3) / 2 + 10 *(error_p3_D1+ error_p3_D2+ error_p3_D3))
    return loss

  def generator_loss(self, Discriminator, real_image, fake_image, use_lsgan=True):
    """  fool discriminator into believing that G(x) is real
    """
    D1_real, D2_real, D3_real = Discriminator(real_image)
    D1_fake, D2_fake, D3_fake =Discriminator(fake_image)
    print("DDDDDDDDDDDDDDDDDDDDDDDDimenton",D1_real[0].shape)
    # relativistic average LSGAN
    error_p1_D1 = tf.reduce_mean(tf.squared_difference(tf.math.subtract(D1_fake[0], tf.reduce_mean(D1_real[0],axis=0)), 1))
    error_p1_D2 = tf.reduce_mean(tf.squared_difference(tf.math.subtract(D2_fake[0], tf.reduce_mean(D2_real[0],axis=0)), 1))
    error_p1_D3 = tf.reduce_mean(tf.squared_difference(tf.math.subtract(D3_fake[0], tf.reduce_mean(D3_real[0],axis=0)), 1))
          
    error_p2_D1 = tf.reduce_mean(tf.squared_difference(D1_real[0], tf.reduce_mean(D1_fake[0],axis=0)))
    error_p2_D2 = tf.reduce_mean(tf.squared_difference(D2_real[0], tf.reduce_mean(D2_fake[0],axis=0)))
    error_p2_D3 = tf.reduce_mean(tf.squared_difference(D3_real[0], tf.reduce_mean(D3_fake[0],axis=0)))
    loss = (error_p1_D1 + error_p1_D2 + error_p1_D3 + error_p2_D1 +  error_p2_D2 +  error_p2_D3) / 2
    return loss


  def VGG19_loss(self, real_image, fake_image):
        loss= tf.zeros(self.batch_size, tf.float32)
        real_image = tf.image.resize_images(real_image, [224, 224])
        fake_image= tf.image.resize_images(fake_image, [224, 224])
        model_real = vgg19.Vgg19(real_image)
        model_fake = vgg19.Vgg19(fake_image)
        feature_real = [model_real.conv1_2, model_real.conv2_2, model_real.conv3_4, model_real.conv4_4, model_real.conv5_4]
        feature_fake = [model_fake.conv1_2, model_fake.conv2_2, model_fake.conv3_4, model_fake.conv4_4, model_fake.conv5_4]
        self.weights = [1.0/(64*(224**2)), 1.0/(128*(112**2)), 1.0/(256*(56**2)), 1.0/(512*(28**2)), 1.0/(512*(14**2))]        

        i=0
        for f, f_ in zip(feature_real, feature_fake):
             loss += tf.reduce_mean(tf.norm(tf.subtract(f, f_), ord=1)*self.weights[i])
             i+=1
        return loss     
  def feature_matching_loss(self, Discriminator, real_image, fake_image):
     
      loss= tf.zeros(self.batch_size, tf.float32)
      D1_o, D2_o, D3_o = Discriminator(real_image)
      D1_fake_o, D2_fake_o, D3_fake_o =Discriminator(fake_image)
      element_layers=[1.0, 1.0/64, 1.0/128,1.0/256,1.0/512]
      for i in range(0,len(D1_o)):
          
        loss += tf.reduce_mean(tf.norm(tf.math.subtract(D1_o[i],D1_fake_o[i]),ord=1)*element_layers[i])
        loss += tf.reduce_mean(tf.norm(tf.math.subtract(D2_o[i],D2_fake_o[i]),ord=1)*element_layers[i])
        loss += tf.reduce_mean(tf.norm(tf.math.subtract(D3_o[i],D3_fake_o[i]),ord=1)*element_layers[i])
      return loss
      
        
#model22 = Pix2Pix_model()middle =
#model22.model()