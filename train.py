#! /usr/bin/env python3

import tensorflow as tf
import numpy as np
import sys
import os
import time
import datetime
import argparse
#import data_helpers
from model import Model
from tensorflow.contrib import learn

# Parameters

args = argparse.ArgumentParser()
args.add_argument("--embedding_dim", type=int, default=300, help="Dimensionality of character embedding (default: 300)")
args.add_argument("--dropout_keep_prob", type=float, default=.5, help="Dropout keep probability (default: 0.5)")

def train():
    with tf.device("/gpu:0"):
        x_test, y = data_helpers.load_data_land_labels(filename,filename)
    text_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(FLAGS.max_sentence_length)
