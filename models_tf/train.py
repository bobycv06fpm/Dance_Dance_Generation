import tensorflow as tf
from model import Pix2Pix_model
#from reader import Reader
from datetime import datetime
import os
import logging
from utils import ImagePool
from reader1 import Reader
from data_generator import *
import glob
from random import shuffle
import numpy as np
import matplotlib.pyplot as plt
FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('batch_size',1, 'batch size, default: 1')
tf.flags.DEFINE_integer('epochs',500, 'batch size, default: 1')
tf.flags.DEFINE_integer('steps', 1000000, 'number of step, default: 1')
tf.flags.DEFINE_integer('G1_epochs', 10, 'number of step for G1 training, default: 1')
tf.flags.DEFINE_integer('image_size', 256, 'image size, default: 256')
tf.flags.DEFINE_bool('use_lsgan', True,
                     'use lsgan (mean squared error) or cross entropy loss, default: True')
tf.flags.DEFINE_string('norm', 'instance',
                       '[instance, batch] use instance norm or batch norm, default: instance')
tf.flags.DEFINE_float('learning_rate', 2e-4,
                      'initial learning rate for Adam, default: 0.0002')
tf.flags.DEFINE_float('beta1', 0.5,
                      'momentum term of Adam, default: 0.5')
tf.flags.DEFINE_float('pool_size', 50,
                      'size of image buffer that stores previously generated images, default: 50')
tf.flags.DEFINE_integer('ngf', 32,
                        'number of gen filters in first conv layer, default: 64')

tf.flags.DEFINE_string('X', '/mnt/sdb2/Dance/data.tfrecords',
                       'X tfrecords file for training, default: /home/jsh/Downloads/facades/data.tfrecords')

tf.flags.DEFINE_string('load_model', "None",
                        'folder of saved model that you wish to continue training (e.g. 20170602-1936), default: None')


def train():
    
  train_image_path = '/home/jsh/Downloads/facades/train'
  image_names = glob.glob(train_image_path +'/*')
  print("yesssssss")
  shuffle(image_names)	
  if FLAGS.load_model is not None:
    checkpoints_dir = "/mnt/sdb6/checkpoint/"+ FLAGS.load_model.lstrip("checkpoints/")
  else:
    current_time = datetime.now().strftime("%Y%m%d-%H%M")
    checkpoints_dir = "/mnt/sdb6/checkpoint/{}".format(current_time)
    try:
      os.makedirs(checkpoints_dir)
    except os.error:
      pass

  graph = tf.Graph()
  with graph.as_default():
    pix2pix_model = Pix2Pix_model(
        X_train_file=FLAGS.X,
        batch_size=FLAGS.batch_size,
        image_size=FLAGS.image_size,
        use_lsgan=FLAGS.use_lsgan,
        norm=FLAGS.norm,
        learning_rate=FLAGS.learning_rate,
        beta1=FLAGS.beta1,
        ngf=FLAGS.ngf
    )
    G_loss, D_loss, fake_image = pix2pix_model.model()
    optimizerG1, optimizerG2, optimizerD  = pix2pix_model.optimize(G_loss, D_loss)
        
    summary_op = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(checkpoints_dir, graph)
    saver = tf.train.Saver()

  with tf.Session(graph=graph) as sess:
    if FLAGS.load_model is not None:
       checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
       meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
       restore = tf.train.import_meta_graph(meta_graph_path)
       restore.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
       step = int(meta_graph_path.split("-")[2].split(".")[0])
    else:
       sess.run(tf.global_variables_initializer())
       step=0
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord) 
   
    
    try:
      
      num_epochs = FLAGS.epochs 
      batch_size =  FLAGS.batch_size
      for epoch in range(0,num_epochs):

        if epoch < FLAGS.G1_epochs:
            optimizer_G = optimizerG1
            print("G11111111111111*************")
        else:
            optimizer_G = optimizerG2
        print("{} epoch: {}".format(datetime.now(), epoch))
        print(len(image_names))
        for iter in np.arange(0,len(image_names),batch_size):
            curr_image_names = image_names[iter*batch_size:(iter+1)*batch_size]
            batch_A,batch_B = load_images_paired(curr_image_names,is_train = True, true_size =256 , enlarge_size = 286)
#                    print(("*********************************************************")
#                    plt.figure()
#                    plt.imshow(batch_A[0])
#                    plt.figure()
#                    plt.imshow(batch_B[0])
#                    plt.show()
            _, _, G_loss_val, D_loss_val, summary = (
                          sess.run(
                              [optimizer_G ,optimizerD, G_loss, D_loss, summary_op],
                               feed_dict={pix2pix_model.real_image: batch_B,
                                  pix2pix_model.label: batch_A}))
#                    summary1 = tf.summary.merge_all()
            train_writer.add_summary(summary, step)
            train_writer.flush()
            
            if step % 100 == 0:
                logging.info('-----------Step %d:-------------' %step)
                logging.info('  G_loss   : {}'.format(G_loss_val))
                logging.info('  D_loss : {}'.format(D_loss_val))
            
            
            if step % 10000 == 0:
                save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
                logging.info("Model saved in file: %s" % save_path)
            step+=1
        save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=epoch)
#                
    except KeyboardInterrupt:
      logging.info('Interrupted')
#      coord.request_stop()
#    except Exception as e:
#      coord.request_stop(e)
#    finally:
#      
#      logging.info("Model saved in file: %s" % save_path)
#      # When done, ask the threads to stop.
#      coord.request_stop()
#      coord.join(threads)
              
def main(unused_argv):
   train()

if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  tf.app.run()
