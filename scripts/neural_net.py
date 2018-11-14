import tensorflow as tf


class NeuralNetwork(object):
  def __init__(self, name, inputs, hidden_layer_sizes, n_outputs, hl_activation,
               final_activation, reuse=False):
    self.name = name
    self.inputs = inputs
    self.hidden_layer_sizes = hidden_layer_sizes
    self.n_outputs = n_outputs
    self.reuse = reuse

    self.variables = []

    # Create the DNN
    with tf.variable_scope(name) as scope:
      if self.reuse:
        scope.reuse_variables()

      previous = self.inputs

      # Create the Hidden Layers
      for i, layer_size in enumerate(hidden_layer_sizes):
        previous, w, b = self._create_layer("Hidden_{}".format(i),
                                            previous, layer_size,
                                            activation=hl_activation)
        self.variables += [w, b]

      # Create the Final Layer
      self.logits, w, b = self._create_layer("Final", previous, n_outputs, activation=final_activation)
      self.variables += [w, b]

  def _create_layer(self, scope_name, inputs, layer_size, activation=None):
    with tf.variable_scope(scope_name) as scope:
      if self.reuse:
        scope.reuse_variables()

      input_size = inputs.get_shape()[1]

      w = tf.get_variable("weights",
                          shape=(input_size, layer_size),
                          dtype=tf.float32)

      b = tf.get_variable("bias",
                          shape=(layer_size,),
                          initializer=tf.zeros_initializer(),
                          dtype=tf.float32)

      output = tf.matmul(inputs, w) + b

      if activation:
        output = activation(output)

      return output, w, b
