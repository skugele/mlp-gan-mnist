import os
from collections import namedtuple
from time import time

import tensorflow as tf
from main import main, FLAGS

TestScenario = namedtuple('TestScenario', ['id',
                                           'hidden_layer_sizes',
                                           'learning_rate',
                                           'n_epochs'])

manifest_file = os.path.join('..', 'train', 'run_manifest.txt')


def run_scenario(scenario):
  tf.reset_default_graph()

  # Set tensorflow run parameters
  FLAGS.train_dir = os.path.join('..', 'train', scenario.id)
  FLAGS.images_dir = os.path.join('..', 'output', 'images', scenario.id)

  FLAGS.disc_layer_sizes = scenario.hidden_layer_sizes
  FLAGS.gen_layer_sizes = scenario.hidden_layer_sizes
  FLAGS.learning_rate = scenario.learning_rate
  FLAGS.n_epochs = scenario.n_epochs

  # Execute Scenario
  start_time = time()
  main()
  elapsed_time = time() - start_time

  # Update manifest
  with open(manifest_file, mode='a+') as fd:
    fd.write('{} [Elapsed Time: {} seconds]'.format(str(scenario), elapsed_time) + '\n')


if __name__ == '__main__':

  try:
    os.remove(manifest_file)
  except OSError:
    pass

  scenario_id = 1
  for n_epochs in [5000, 10000, 25000]:
    for hidden_layer_sizes in [[128], [256], [512], [64, 32], [128, 64]]:
      for learning_rate in [0.01, 0.001, 0.0001]:
        scenario = TestScenario(id=str(scenario_id),
                                hidden_layer_sizes=hidden_layer_sizes,
                                learning_rate=learning_rate,
                                n_epochs=n_epochs)

        try:
          run_scenario(scenario)
        except Exception as e:
          print('Caught Exception During Run: ' + e.message)

        scenario_id += 1
