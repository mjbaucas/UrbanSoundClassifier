import glob
import os
import math
import librosa
import numpy as np
import sys
from statistics import mode
import time

import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support, classification_report

def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    X[X == np.nan] = 0.0
    X[X == np.inf] = 10.0
    #print("{} >>> {}".format(np.isnan(X), np.isinf(X)))
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz

def parse_audio_file(file_dir):
    features, labels = np.empty((0,193)), np.empty(0)
    try:
        mfccs, chroma, mel, contrast,tonnetz = extract_feature(file_dir)
        ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
        features = np.vstack([features,ext_features])
        labels = np.append(labels, 1)
    except Exception as e:
        print(e)
    return np.array(features), np.array(labels, dtype = np.int)

def encode_matrix(labels, num_unique_labels):
    num_labels = len(labels)
    encoded_matrix = np.zeros((num_labels, num_unique_labels), dtype=int)
    encoded_matrix[np.arange(num_labels), labels] = 1
    return encoded_matrix

if __name__ == "__main__":
    training_steps = 2500
    num_input = 193
    num_hidden_one = 280 
    num_hidden_two = 300
    learning_rate = 0.2
    num_classes = 2

    test_x, test_y = parse_audio_file(os.getcwd() + "/" + sys.argv[1])
    test_y = encode_matrix(test_y, num_classes)
    X = tf.placeholder(tf.float32,[None,num_input])
    Y = tf.placeholder(tf.float32,[None,num_classes])

    weights_h1 = tf.Variable(tf.random_normal([num_input,num_hidden_one]))
    weights_h2 = tf.Variable(tf.random_normal([num_hidden_one,num_hidden_two]))
    weights_out = tf.Variable(tf.random_normal([num_hidden_two,num_classes]))

    bias_b1 = tf.Variable(tf.random_normal([num_hidden_one]))
    bias_b2 = tf.Variable(tf.random_normal([num_hidden_two]))
    bias_out = tf.Variable(tf.random_normal([num_classes]))

    first_layer = tf.add(tf.matmul(X, weights_h1), bias_b1)
    second_layer = tf.add(tf.matmul(first_layer, weights_h2), bias_b2)
    out_layer = tf.matmul(second_layer, weights_out) + bias_out
    prediction = tf.nn.softmax(out_layer)

    y_true, y_pred = [], []
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        
        for i in range(0,100):
            sess.run(init)
        
            saver = tf.train.import_meta_graph('Models/DogBarking_training_70_2500_2.ckpt.meta')
            saver.restore(sess, "Models/DogBarking_training_70_2500_2.ckpt")

            y_pred.append(sess.run(tf.argmax(prediction,1),feed_dict={X: test_x})[0])
            y_true.append(sess.run(tf.argmax(test_y,1))[0])

    print(y_pred)
    print(classification_report(y_true, y_pred))

