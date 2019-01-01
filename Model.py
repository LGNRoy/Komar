import tensorflow as tf
import numpy as np
import os

SEQ_LEN = 10 
BATCH_SIZE = 4 
LEFT_CONTEXT = 5

HEIGHT = 480
WIDTH = 640
CHANNELS = 3 # RGB

slim = tf.contrib.slim

layer_norm = lambda x: tf.contrib.layers.layer_norm(inputs=x, center=True, scale=True, activation_fn=None, trainable=True)

def get_optimizer(loss, lrate):
    optimizer = tf.train.AdamOptimizer(learning_rate=lrate)
    gradvars = optimizer.compute_gradients(loss)
    gradients, v = zip(*gradvars)
    print ([x.name for x in v])
    gradients, _ = tf.clip_by_global_norm(gradients, 15.0)
    return optimizer.apply_gradients(zip(gradients, v))

def apply_vision_simple(image, keep_prob, batch_size, seq_len, scope=None, reuse=None):
    video = tf.reshape(image, shape=[batch_size, LEFT_CONTEXT + seq_len, HEIGHT, WIDTH, CHANNELS])
    with tf.variable_scope(scope, 'Vision', [image], reuse=reuse):
        net = slim.convolution(video, num_outputs=64, kernel_size=[3,12,12], stride=[1,6,6], padding="VALID")
        net = tf.nn.dropout(x=net, keep_prob=keep_prob)
        aux1 = slim.fully_connected(tf.reshape(net[:, -seq_len:, :, :, :], [batch_size, seq_len, -1]), 128, activation_fn=None)
        
        net = slim.convolution(net, num_outputs=64, kernel_size=[2,5,5], stride=[1,2,2], padding="VALID")
        net = tf.nn.dropout(x=net, keep_prob=keep_prob)
        aux2 = slim.fully_connected(tf.reshape(net[:, -seq_len:, :, :, :], [batch_size, seq_len, -1]), 128, activation_fn=None)
        
        net = slim.convolution(net, num_outputs=64, kernel_size=[2,5,5], stride=[1,1,1], padding="VALID")
        net = tf.nn.dropout(x=net, keep_prob=keep_prob)
        aux3 = slim.fully_connected(tf.reshape(net[:, -seq_len:, :, :, :], [batch_size, seq_len, -1]), 128, activation_fn=None)
        
        net = slim.convolution(net, num_outputs=64, kernel_size=[2,5,5], stride=[1,1,1], padding="VALID")
        net = tf.nn.dropout(x=net, keep_prob=keep_prob)
        # at this point the tensor 'net' is of shape batch_size x seq_len x ...
        aux4 = slim.fully_connected(tf.reshape(net, [batch_size, seq_len, -1]), 128, activation_fn=None)
        
        net = slim.fully_connected(tf.reshape(net, [batch_size, seq_len, -1]), 1024, activation_fn=tf.nn.relu)
        net = tf.nn.dropout(x=net, keep_prob=keep_prob)
        net = slim.fully_connected(net, 512, activation_fn=tf.nn.relu)
        net = tf.nn.dropout(x=net, keep_prob=keep_prob)
        net = slim.fully_connected(net, 256, activation_fn=tf.nn.relu)
        net = tf.nn.dropout(x=net, keep_prob=keep_prob)
        net = slim.fully_connected(net, 128, activation_fn=None)
        return layer_norm(tf.nn.elu(net + aux1 + aux2 + aux3 + aux4)) # aux[1-4] are residual connections (shortcuts)

class SamplingRNNCell(tf.nn.rnn_cell.RNNCell):
  """Simple sampling RNN cell."""

  def __init__(self, num_outputs, use_ground_truth, internal_cell):
    """
    if use_ground_truth then don't sample
    """
    self._num_outputs = num_outputs
    self._use_ground_truth = use_ground_truth # boolean
    self._internal_cell = internal_cell # may be LSTM or GRU or anything
  
  @property
  def state_size(self):
    return self._num_outputs, self._internal_cell.state_size # previous output and bottleneck state

  @property
  def output_size(self):
    return self._num_outputs # steering angle, torque, vehicle speed

  def __call__(self, inputs, state, scope=None):
    (visual_feats, current_ground_truth) = inputs
    prev_output, prev_state_internal = state
    context = tf.concat(1, [prev_output, visual_feats])
    new_output_internal, new_state_internal = internal_cell(context, prev_state_internal) # here the internal cell (e.g. LSTM) is called
    new_output = tf.contrib.layers.fully_connected(
        inputs=tf.concat(1, [new_output_internal, prev_output, visual_feats]),
        num_outputs=self._num_outputs,
        activation_fn=None,
        scope="OutputProjection")
    # if self._use_ground_truth == True, we pass the ground truth as the state; otherwise, we use the model's predictions
    return new_output, (current_ground_truth if self._use_ground_truth else new_output, new_state_internal)
    
    