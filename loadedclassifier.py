import glob
import os
import math
import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram

import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support, classification_report

def extract_feature(file_name, sample_rate):
    X = np.loadtxt(file_name)
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

def parse_audio_files(parent_dir,sub_dirs,file_ext='*.txt'):
    features, labels = np.empty((0,193)), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            try:
                mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn, 22050)
                ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
                features = np.vstack([features,ext_features])
                labels = np.append(labels, 1)
            except Exception as e:
                print(e)
    return np.array(features), np.array(labels, dtype = np.int)

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = 10
    one_hot_encode = np.zeros((n_labels,n_unique_labels), dtype=int)
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode

if __name__ == "__main__":
    parent_dir = 'SoundfilesB'
    
    test_sub_dirs = ['3']
    test_x, test_y = parse_audio_files(parent_dir,test_sub_dirs)
    test_y = one_hot_encode(test_y)

    training_steps = 2500
    num_input = 193
    num_hidden_one = 280 
    num_hidden_two = 300
    learning_rate = 0.01
    num_classes = 2
    
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

    y_true, y_pred = None, None

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, "Models/DogBarking_training_70_2500.ckpt")
        
        y_pred = sess.run(tf.argmax(prediction,1),feed_dict={X: test_x})
        y_true = sess.run(tf.argmax(test_y,1))

    p,r,f,s = precision_recall_fscore_support(y_true, y_pred, average='micro')

    print("Precision:", round(p,3))
    print("Recall:", round(r,3))
    print("F-Score:", round(f,3))
    if s is not None:
        print("Support:", round(s,3))

    print(classification_report(y_true, y_pred))