#! /usr/bin/env python
import os

import numpy as np
import tensorflow as tf

from data_utils import get_mnist_datasets, write_images_plot
from neural_net import NeuralNetwork

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '../train', 'Directory where to write event logs and checkpoints.')
tf.app.flags.DEFINE_string('data_dir', '../data', 'Directory where to store/read training data.')
tf.app.flags.DEFINE_string('images_dir', '../output/images', 'Directory where sampled images will be written.')

tf.app.flags.DEFINE_string('images_prefix', 'image_plot', 'The prefix to add to sampled images files.')
tf.app.flags.DEFINE_string('images_file_ext', 'png', 'The file extension to use for sampled images files.')
tf.app.flags.DEFINE_integer('images_out_frequency', '500', 'Epochs between sampled image writes.')

# Flags for logging
tf.app.flags.DEFINE_boolean('log_device_placement', False, 'Whether to log device placement.')
tf.app.flags.DEFINE_integer('log_frequency', 10, 'How often to log results to the console.')

# Flags for creation of computational graph
tf.app.flags.DEFINE_list('gen_layer_sizes', [128, 64], 'Layer sizes for the generator network.')
tf.app.flags.DEFINE_list('disc_layer_sizes', [128, 64], 'Layer sizes for the discriminiator network.')

# Flags for termination criteria
tf.app.flags.DEFINE_integer('n_epochs', 1000000, 'Max number of training epochs.')

# Flags for algorithm parameters
tf.app.flags.DEFINE_float('learning_rate', 0.0003, 'The learning rate (eta) to be used for neural networks.')
tf.app.flags.DEFINE_integer('batch_size', 64, 'Mini-batch size from training.')

# Input dimensions
tf.app.flags.DEFINE_integer('image_dim', 784, 'Dimensions used for real image vector.')
tf.app.flags.DEFINE_integer('noise_dim', 100, 'Dimensions used for noise vector (z).')

# Suppresses most informational messages/warnings
tf.logging.set_verbosity(tf.logging.ERROR)


class GAN(object):
  def __init__(self, gen_layers, disc_layers, learning_rate, image_dim, noise_dim):
    self.global_step = tf.train.get_or_create_global_step()

    self.images = tf.placeholder(name='images', shape=(None, image_dim), dtype=tf.float32)
    self.noise = tf.placeholder(name='noise', shape=(None, noise_dim), dtype=tf.float32)

    # Create discriminator and generator neural networks
    self.generator = NeuralNetwork(name='Generator',
                                   inputs=self.noise,
                                   hidden_layer_sizes=gen_layers,
                                   n_outputs=image_dim,
                                   hl_activation=tf.nn.relu,
                                   final_activation=None)

    self.discriminator_real = NeuralNetwork(name='Discriminator',
                                            inputs=self.images,
                                            hidden_layer_sizes=disc_layers,
                                            n_outputs=1,
                                            hl_activation=tf.nn.leaky_relu,
                                            final_activation=None,
                                            reuse=False)

    self.discriminator_fake = NeuralNetwork(name='Discriminator',
                                            inputs=self.generator.logits,
                                            hidden_layer_sizes=disc_layers,
                                            n_outputs=1,
                                            hl_activation=tf.nn.leaky_relu,
                                            final_activation=None,
                                            reuse=True)

    # Uses "one-sided label smoothing" (from tips and tricks in NIPS 2016 Tutorial on GANs)
    self.d_loss_real = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(labels=0.9 * tf.ones_like(self.discriminator_real.logits),
                                              logits=self.discriminator_real.logits))
    self.d_loss_fake = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(labels=0.1 * tf.ones_like(self.discriminator_fake.logits),
                                              logits=self.discriminator_fake.logits))

    self.d_loss = self.d_loss_real + self.d_loss_fake
    self.g_loss = (tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.discriminator_fake.logits),
                                              logits=self.discriminator_fake.logits)))

    # Training operation (momentum beta has been adjusted to 0.5 based upon comments in "UNSUPERVISED
    # REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS", Radford et al
    # though it is not clear whether the parameters that work well with a CNN will be the same
    # for non-convolutional neural network.
    self.d_train = (tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5)
                    .minimize(loss=self.d_loss,
                              var_list=self.discriminator_fake.variables))

    self.g_train = (tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5)
                    .minimize(loss=self.g_loss,
                              global_step=self.global_step,
                              var_list=self.generator.variables))


def train():
  model = GAN(gen_layers=FLAGS.gen_layer_sizes,
              disc_layers=FLAGS.disc_layer_sizes,
              learning_rate=FLAGS.learning_rate,
              image_dim=FLAGS.image_dim,
              noise_dim=FLAGS.noise_dim)

  data = get_mnist_datasets(FLAGS.data_dir)

  tf.summary.scalar("global_step", model.global_step)
  tf.summary.scalar("discriminator_loss", model.d_loss)
  tf.summary.scalar("generator_loss", model.g_loss)

  with tf.train.SingularMonitoredSession(
      # save/load model state
      checkpoint_dir=FLAGS.train_dir,
      hooks=[tf.train.StopAtStepHook(last_step=FLAGS.n_epochs),
             tf.train.NanTensorHook(model.d_loss),
             tf.train.NanTensorHook(model.g_loss),
             tf.train.CheckpointSaverHook(
               checkpoint_dir=FLAGS.train_dir,
               save_steps=1000,
               saver=tf.train.Saver()),
             tf.train.SummarySaverHook(
               save_steps=100,
               output_dir=FLAGS.train_dir,
               scaffold=tf.train.Scaffold(summary_op=tf.summary.merge_all()),
             )],

      # Can be configured for multi-machine training (check out docs for this class)
      config=tf.ConfigProto(
        log_device_placement=FLAGS.log_device_placement)) as mon_sess:

    while not mon_sess.should_stop():

      # Train Discriminator
      batch_images, _ = data.train.next_batch(FLAGS.batch_size)
      noise_sample = np.random.uniform(-1.0, 1.0, size=(FLAGS.batch_size, FLAGS.noise_dim)).astype(np.float32)
      step, d_loss_val, _ = mon_sess.run([model.global_step, model.d_loss, model.d_train],
                                         feed_dict={model.noise: noise_sample,
                                                    model.images: batch_images})

      if not mon_sess.should_stop():
        # Train Generator
        noise_sample = np.random.uniform(-1.0, 1.0, size=(FLAGS.batch_size, FLAGS.noise_dim)).astype(np.float32)
        step, g_loss_val, _ = mon_sess.run([model.global_step, model.g_loss, model.g_train],
                                           feed_dict={model.noise: noise_sample,
                                                      model.images: batch_images})

      if step % FLAGS.images_out_frequency == 0:
        noise_sample = np.random.uniform(-1.0, 1.0, size=(16, FLAGS.noise_dim)).astype(np.float32)
        images = mon_sess.raw_session().run(model.generator.logits,
                                            feed_dict={model.images: batch_images,
                                                       model.noise: noise_sample})

        write_images_plot(images=images,
                          img_width=28, img_height=28,
                          rows=4, cols=4,
                          filepath=os.path.join(FLAGS.images_dir,
                                                '{}_{}.{}'.format(
                                                  FLAGS.images_prefix,
                                                  str(step),
                                                  FLAGS.images_file_ext)))


def main(argv=None):
  if not tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.MakeDirs(FLAGS.train_dir)

  if not tf.gfile.Exists(FLAGS.images_dir):
    tf.gfile.MakeDirs(FLAGS.images_dir)

  if not tf.gfile.Exists(FLAGS.data_dir):
    tf.gfile.MakeDirs(FLAGS.data_dir)

  train()


if __name__ == '__main__':
  tf.app.run()
