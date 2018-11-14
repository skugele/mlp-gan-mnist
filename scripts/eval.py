import ast
import os
import re

import numpy as np
import tensorflow as tf

from data_utils import get_mnist_datasets
from main import GAN, FLAGS

manifest_file = os.path.join('..', 'train', 'run_manifest.txt')
eval_file = os.path.join('..', 'output', 'eval', 'disc_perf.txt')

data = get_mnist_datasets(FLAGS.data_dir)

real_images, _ = data.test.next_batch(256)
noise_sample = np.random.uniform(-1.0, 1.0, size=(256, FLAGS.noise_dim)).astype(np.float32)

with open(manifest_file, mode='r') as infile:
  with open(eval_file, mode='a+') as outfile:
    for line in infile:
      g = tf.Graph()
      with tf.Session(graph=g) as sess:
        scenario_id = re.search('id=\'(.*)\',', line).group(1)
        hidden_layer_sizes = ast.literal_eval(re.search('hidden_layer_sizes=(\[.*\]),', line).group(1))
        learning_rate = re.search('learning_rate=(.+),', line).group(1)

        model = GAN(gen_layers=hidden_layer_sizes,
                    disc_layers=hidden_layer_sizes,
                    learning_rate=learning_rate,
                    image_dim=FLAGS.image_dim,
                    noise_dim=FLAGS.noise_dim)

        ckpt_path = os.path.join('..', 'train', scenario_id)
        ckpt = tf.train.latest_checkpoint(ckpt_path)

        saver = tf.train.Saver()
        saver.restore(sess, ckpt)

        mean_prob_fake_or_real = tf.reduce_mean(tf.nn.sigmoid(model.discriminator_real.logits))

        # Probability on real images
        p_real = sess.run(mean_prob_fake_or_real,
                          feed_dict={model.images: real_images,
                                     model.noise: noise_sample})

        fake_images = sess.run(model.generator.logits,
                               feed_dict={model.images: real_images,
                                          model.noise: noise_sample})

        # Probability on fake images
        p_fake = sess.run(mean_prob_fake_or_real,
                          feed_dict={model.images: fake_images,
                                     model.noise: noise_sample})

        outfile.write(','.join([scenario_id, str(p_real), str(p_fake)]) + '\n')


