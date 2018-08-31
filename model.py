import tensorflow as tf

class Model(object):
    def __init__(
            self, sequence_length, num_classes, vocab_size,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        self.input_left = tf.placeholder(tf.int32, [None, sequence_length], name='input_left')
        self.input_right = tf.placeholder(tf.int32, [None, sequence_length], name='input_right')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/gpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name='W_text')
            self.embedded_chars = tf.nn.embedding_lookup(self.W_text, self.input_text)
        pass
