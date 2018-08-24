#! /usr/bin/env python3

import tensorflow as tf
import numpy as np
import sys
import os
import time
import datetime
import argparse
import data_helpers
from rnn import RNN
from tensorflow.contrib import learn

# Parameters
# ==================================================

# Data loading params
args = argparse.ArgumentParser()
args.add_argument("--dev_sample_percentage", type=float, default=.1, help="Percentage of the training data to use for validation")
args.add_argument("--positive_data_file", type=str, default="./data/rt-polaritydata/rt-polarity.pos", help="Data source for the positive data.")
args.add_argument("--negative_data_file", type=str, default="./data/rt-polaritydata/rt-polarity.neg", help="Data source for the negative data.")
args.add_argument("--max_sentence_length", type=int, default=100, help="Max sentence length in train/test data (Default: 100)")

# Model Hyperparameters
args.add_argument("--cell_type", type=str, default="lstm", help="Type of rnn cell. Choose 'vanila', 'gru', or 'lstm' (Default: lstm)")
args.add_argument("--word2vec", type=str, default=None, help="Word2vec file with pre-trained embeddings")
args.add_argument("--embedding_dim", type=int, default=300, help="Dimensionality of character embedding (default: 300)")
args.add_argument("--hidden_size", type=int, default=128, help="Dimensionality of character embedding (default: 128)")
args.add_argument("--dropout_keep_prob", type=float, default=.5, help="Dropout keep probability (default: 0.5)")
args.add_argument("--l2_reg_lambda", type=float, default=.0, help="L2 regularization labda (default: 0.0)")

# Training Parameters
args.add_argument("--batch_size", type=int, default=64, help="Batch Size (default: 64)")
args.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs (default:100)")
args.add_argument("--display_every", type=int, default=10, help="Number of iterations to display training info.")
args.add_argument("--evaluate_every", type=int, default=100, help="Evaluate model on dev set after this many steps (default: 100)")
args.add_argument("--checkpoint_every", type=int, default=100, help="Save model after this many steps (default: 100)")
args.add_argument("--num_checkpoints", type=int, default=5, help="Number of checkpoints to store (default: 5)")
args.add_argument("--learning_rate", type=float, default=1e-3, help="Which learning rate to start with. (Default: 1e-3)")

# Misc Parameters
args.add_argument("--allow_soft_placement", type=bool, default=True, help="Allow device soft device placement")
args.add_argument("--log_device_placement", type=bool, default=False, help="Log placement of ops on devices")

FLAGS = args.parse_args()

def train():
    with tf.device('/gpu:0'):
        x_test, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
    text_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(FLAGS.max_sentence_length)
    x = np.array(list(text_vocab_processor.fit_transform(x_test)))
    print("Text Vocabulary Size: {:d}".format(len(text_vocab_processor.vocabulary_)))

    print("x = {0}".format(x.shape))
    print("y = {0}".format(y.shape))
    print('')

    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    #Split train/test set
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    print("Train/Dev split: {:d}/{:d}\n".format(len(y_train), len(y_dev)))

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            rnn = RNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(text_vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                cell_type=FLAGS.cell_type,
                hidden_size=FLAGS.hidden_size,
                l2_reg_lambda=FLAGS.l2_reg_lambda
            )

            # Define Training Procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(rnn.loss, global_step=global_step)
            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", rnn.loss)
            acc_summary = tf.summary.scalar("accuracy", rnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev Summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            text_vocab_processor.save(os.path.join(out_dir, "text_vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # Pre-trained word2vec
            if FLAGS.word2vec:
                # initial matrix with random uniform
                initW = np.random.uniform(-0.25, 0.25, (len(text_vocab_processor.vocabulary_), FLAGS.embedding_dim))
                # load any vectors from the word2vec
                print("Load word2vec file {0}".format(FLAGS.word2vec))
                with open(FLAGS.word2vec, "rb") as f:
                    header = f.readline()
                    vocab_size, layer1_size = map(int, header.split())
                    binary_len = np.dtype("float32").itemsize * layer1_size
                    for line in range(vocab_size):
                        word = []
                        while True:
                            ch = f.read(1).decode('latin-1') # OR utf8
                            if ch == ' ':
                                word = ''.join(word)
                                break
                            if ch != '\n':
                                word.append(ch)
                            idx = text_vocab_processor.vocabulary_.get(word)
                            if idx != 0:
                                initW[idx] = np.fromstring(f.read(binary_len), dtype='float32')
                            else:
                                f.read(binary_len)
                sess.run(rnn.W_text.assign(initW))
                print("Success to load pre-trained word2vec model!\n")
            # Generate bathes
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                # Train
                feed_dict = {
                    rnn.input_text: x_batch,
                    rnn.input_y: y_batch,
                    rnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, rnn.loss, rnn.accuracy], feed_dict)
                train_summary_writer.add_summary(summaries, step)

                # Training log display
                if step % FLAGS.display_every == 0:
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

                # Evaluation
                if step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    feed_dict_dev = {
                        rnn.input_text: x_dev,
                        rnn.input_y: y_dev,
                        rnn.dropout_keep_prob: 1.0
                    }
                    summaries_dev, loss, accuracy = sess.run(
                        [dev_summary_op, rnn.loss, rnn.accuracy], feed_dict_dev)
                    dev_summary_writer.add_summary(summaries_dev, step)

                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}\n".format(time_str, step, loss, accuracy))

                # Model checkpoint
                if step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=step)
                    print("Saved model checkpoint to {}\n".format(path))


def main(argv):
    train()

if __name__=='__main__':
    main(sys.argv)