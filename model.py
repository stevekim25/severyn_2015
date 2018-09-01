import tensorflow as tf

class Model(object):
    def __init__(
            self, left_length, right_length, num_classes, vocab_size,
            embedding_size, filter_sizes, num_filters, num_hidden, l2_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        self.input_left = tf.placeholder(tf.int32, [None, left_length], name='input_left') # or sequence_length both.
        self.input_right = tf.placeholder(tf.int32, [None, right_length], name='input_right')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.input_y = tf.placeholder(tf.int32, [None,num_classes], name='input_y')

        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/gpu:0'), tf.name_scope("embedding"):
            self.W_text = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name='W_text')

            self.embedded_left_chars = tf.nn.embedding_lookup(self.W_text, self.input_left)
            self.embedded_left_chars_expanded = tf.expand_dims(self.embedded_left_chars, -1)
            self.embedded_right_chars = tf.nn.embedding_lookup(self.W_text, self.input_right)
            self.embedded_right_chars_expanded = tf.expand_dims(self.embedded_right_chars, -1)

        pooled_left_outputs = []
        pooled_right_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-left-%s" % filter_size):
                # Convolution Layer for left
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
                conv_left = tf.nn.conv2d(
                    self.embedded_left_chars_expanded,
                    W, strides=[1,1,1,1], padding='VALID', name='conv_left')
                h_left = tf.nn.relu(tf.nn.bias_add(conv_left,b), name='relu')

                # Maxpoolling over the outputs
                pool_left = tf.nn.max_pool(
                    h_left,
                    ksize=[1, left_length - filter_size + 1, 1,1],
                    strides=[1,1,1,1], padding='VALID', name='pool_left')
                pooled_left_outputs.append(pool_left)

            with tf.name_scope("conv-maxpool-right-%s" % filter_size):
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
                conv_right = tf.nn.conv2d(
                    self.embedded_right_chars_expanded,
                    W, strides=[1,1,1,1], padding='VALID', name='conv_right')
                h_right = tf.nn.relu(tf.nn.bias_add(conv_right,b), name='relu')

                pool_right = tf.nn.max_pool(
                    h_right,
                    ksize=[1, right_length - filter_size + 1, 1,1],
                    strides=[1,1,1,1], padding='VALID', name='pool_right')
                pooled_right_outputs.append(pool_right)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool_left = tf.reshape(tf.concat(pooled_left_outputs,3), shape=[-1, num_filters_total], name='h_pool_left')
        self.h_pool_right = tf.reshape(tf.concat(pooled_right_outputs,3), [-1, num_filters_total], name='h_pool_right')

        with tf.name_scope("similarity"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_filters_total])    # initializer?
            self.transform_left = tf.matmul(self.h_pool_left, W)
            self.sims = tf.reduce_sum(tf.multiply(self.transform_left, self.h_pool_right),1, keep_dims=True)
            print(self.sims)

        self.new_input = tf.concat([self.h_pool_left, self.sims, self.h_pool_right],1, name='new_input')
        with tf.name_scope("hidden"):
            W = tf.get_variable(
                "W_hidden",
                shape=[2*num_filters_total+1, num_hidden])
            b = tf.Variable(tf.constant(0.1, shape=[num_hidden]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.hidden_output = tf.nn.relu(tf.nn.xw_plus_b(self.new_input, W, b, name='hidden_output'))
        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.hidden_output, self.dropout_keep_prob, name='hidden_output_drop')
            print(self.h_drop)
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W_output",
                shape=[num_hidden,2],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[2]), name='b')    #2: classification, 1: regression
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name='scores')
            self.predictions = tf.argmax(self.scores, 1, name='predictions')   # for classification

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name='accuracy')
