# -*- coding: utf-8 -*-
'''
A Dynamic Recurrent Neural Network (LSTM) implementation example using
TensorFlow library. This example is using a toy dataset to classify linear
sequences. The generated sequences have variable length.
Long Short Term Memory paper: http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
import random
import reader
import numpy

# ====================
#  TOY DATA PADDING
# ====================
class ToySequenceData(object):
    """ Generate sequence of data with dynamic length.
    This class generate samples for training:
    - Class 0: linear sequences (i.e. [0, 1, 2, 3,...])
    - Class 1: random sequences (i.e. [1, 3, 10, 7,...])
    NOTICE:
    We have to pad each sequence to reach 'max_seq_len' for TensorFlow
    consistency (we cannot feed a numpy array with inconsistent
    dimensions). The dynamic calculation will then be perform thanks to
    'seqlen' attribute that records every actual sequence length.
    """
    def __init__(self, start=0,n_samples=10000, max_seq_len=200):
        self.data = []
        self.labels = []
        self.seqlen = []
        x, y = reader.load()
        for i in range(start,n_samples):
            s = x[i]
            length = len(x[i])
            # Monitor sequence length for TensorFlow dynamic calculation
            if length >200:
                self.seqlen.append(200)
            else:
                self.seqlen.append(length)
            if length >= max_seq_len:
                s=x[i][:max_seq_len]
            else:
                # Use last point to pad sequence for dimension consistency
                s += [x[i][length-1] for j in range(max_seq_len - length)]
            self.data.append(s)
            if y[i][0][0] == 1.:
                self.labels.append([1., 0.])
            else:
                self.labels.append([0., 1.])
        self.batch_id = 0
        self.perm = numpy.random.permutation(numpy.arange(len(self.data)))

    def next(self, batch_size):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        if self.batch_id == len(self.data):
            self.batch_id = 0
            self.perm = numpy.random.permutation(numpy.arange(len(self.data)))
        batch_data=[]
        batch_labels=[]
        batch_seqlen=[]
        for i in self.perm[self.batch_id:min(self.batch_id+batch_size, len(self.data))]:
            batch_data.append(self.data[i])
            batch_labels.append(self.labels[i])
            batch_seqlen.append(self.seqlen[i])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data, batch_labels, batch_seqlen

class ToySequenceTest(object):
  def __init__(self, start=0, n_samples=100000, max_seq_len=200):
    self.data = []
    self.seqlen = []
    x = reader.loadTest()
    for i in range(start, n_samples):
        s = x[i]
        length = len(x[i])
        # Monitor sequence length for TensorFlow dynamic calculation
        if length > 200:
            self.seqlen.append(200)
        else:
            self.seqlen.append(length)
        if length >= max_seq_len:
            s = x[i][:max_seq_len]
        else:
            # Use last point to pad sequence for dimension consistency
            s += [x[i][length - 1] for j in range(max_seq_len - length)]
        self.data.append(s)

# ==========
#   MODEL
# ==========

# Parameters
training_iters = 1000000
batch_size = 128
display_step = 10

# Network Parameters
learning_rate=0.001
seq_max_len = 200 # Sequence max length
n_hidden = 128 # hidden layer num of features
n_classes = 2 # linear sequence or not

trainset = ToySequenceData()
testset = ToySequenceTest()

# tf Graph input
x = tf.placeholder("float", [None, seq_max_len, 3])
y = tf.placeholder("float", [None, n_classes])
dropout = tf.placeholder("float")
# A placeholder for indicating each sequence length
seqlen = tf.placeholder(tf.int32, [None])

# Define weights
weights = {
    'out': tf.Variable(tf.truncated_normal([2*n_hidden, n_classes], stddev=0.1))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def cnn(x,seqlen,num_filters=20,filter_size =3):
    # Add CNN get filters and combine with word
    filter_shape = [filter_size, 3, num_filters]
    W_conv = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_conv")
    # b_conv = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b_conv")

    conv = tf.nn.conv1d(x,
                        W_conv,
                        stride=1,
                        padding="SAME",
                        name="conv")  # will have dimensions [batch_size,out_width,num_filters] out_width is a function of max_words,filter_size and stride_size #(?, 3051, 20)
    # out_width for same padding with stride 1  given by (sequence_length)
    # print("conv.get_Shape(): ", conv.get_shape())
    # Apply nonlinearity
    # h = tf.nn.bias_add(conv, b_conv,name="add bias")#does not change dimensions
    h_expand = tf.expand_dims(conv, -1)
    print("h_expand.get_Shape(): ", h_expand.get_shape())
    pooled = tf.nn.max_pool(
        h_expand,
        # [batch, height, width, channels]
        ksize=[1, seqlen, 1, 1],
        # On the batch size dimension and the channels dimension, ksize is 1 because we don't want to take the maximum over multiple examples, or over multiples channels.
        strides=[1, 1, 1, 1],
        padding='SAME',
        name="pooled")
    # print("pooled.get_Shape(): ",pooled.get_shape())
    # [batch_size,(sequence_length), num_filters, 1] --> [batch, sequence_length, num_filters] , same as word_embedding layer (?, 113, 20, 1) --> (?, 113, 20)
    char_pool_flat = tf.reshape(pooled, [-1, seqlen, num_filters], name="char_pool_flat")
    # print("self.char_pool_flat.get_shape(): ",self.char_pool_flat.get_shape())
    # [batch, sequence_length, word_embedd_dim+num_filters]
    word_char_features = tf.concat([x, char_pool_flat],
                                        axis=2)  # we mean that the feature with index 2 i/e num_filters is variable
    # print("self.word_char_features.get_shape(): ",self.word_char_features.get_shape())
    word_char_features_dropout = tf.nn.dropout(word_char_features, 0.9,name="word_char_features_dropout")
    return word_char_features_dropout

def dynamicRNN(x, seqlen, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, seq_max_len, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
    lstm_cell_bk = tf.contrib.rnn.BasicLSTMCell(n_hidden)
    lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=dropout)
    lstm_cell_bk = tf.contrib.rnn.DropoutWrapper(lstm_cell_bk, output_keep_prob=dropout)
    # Get lstm cell output, providing 'sequence_length' will perform dynamic
    # calculation.


    #outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32,
    #                            sequence_length=seqlen)
    outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_cell, lstm_cell_bk, x,
                                                 dtype=tf.float32)


    # When performing dynamic calculation, we must retrieve the last
    # dynamically computed output, i.e., if a sequence length is 10, we need
    # to retrieve the 10th output.
    # However TensorFlow doesn't support advanced indexing yet, so we build
    # a custom op that for each sample in batch size, get its length and
    # get the corresponding relevant output.

    # 'outputs' is a list of output at every timestep, we pack them in a Tensor
    # and change back dimension to [batch_size, n_step, n_input]
    outputs = tf.stack(outputs)
    outputs = tf.transpose(outputs, [1, 0, 2])

    # Hack to build the indexing and retrieve the right output.
    batch_size = tf.shape(outputs)[0]
    # Start indices for each sample
    index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)
    # Indexing
    outputs = tf.gather(tf.reshape(outputs, [-1, 2*n_hidden]), index)

    # Linear activation, using outputs computed above
    return tf.matmul(outputs, weights['out']) + biases['out']

cnn_x = cnn(x,200)
pred = dynamicRNN(cnn_x, seqlen, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    best = 0
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y, batch_seqlen = trainset.next(batch_size)

        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,dropout:0.9,
                                       seqlen: batch_seqlen})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y,dropout:0.9,
                                                seqlen: batch_seqlen})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y,dropout:0.9,
                                             seqlen: batch_seqlen})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
            if acc >= 0.99:
                best = best+1
            if acc < 0.99:
                best = 0
        step += 1
        if best > 15:
            break
    print("Optimization Finished!")

    # Calculate test
    test_data = testset.data
    test_seqlen = testset.seqlen
    p = sess.run(pred, feed_dict={x: test_data,dropout:1.0,seqlen: test_seqlen})
    with open('result.txt', 'w') as f:
        for i,label in enumerate(p):
            if label[0]<label[1]:
                f.write(str(i+1))
                f.write('\n')
