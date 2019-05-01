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

def extract_features(file_name, sample_rate):
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

def parse_audio_file(file_dir):
    features, labels = np.empty((0,193)), np.empty(0)
    try:
        mfccs, chroma, mel, contrast,tonnetz = extract_feature(file_dir)
        ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
        features = np.vstack([features,ext_features])
        labels = np.append(labels, 1)
        labels = np.append(labels, 0)
    except Exception as e:
        print(e)
    return np.array(features), np.array(labels, dtype = np.int)

def parse_audio_files(parent_dir, sub_dirs, percent_split, file_ext='*.txt'):
    training_dir = "{}/TrainingSet/{}%_{}".format(parent_dir, str(percent_split), "_".join(sub_dirs))
    if not os.path.exists(training_dir):
         os.makedirs(training_dir)
    print(training_dir)
    features, labels = np.empty((0,193)), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            sound_name_split = fn.split('/')[-1].split('.')[0]
            if not os.path.isfile("{}/{}.txt".format(training_dir, sound_name_split)):
                print(fn)
                try:
                    #mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn)
                    mfccs, chroma, mel, contrast,tonnetz = extract_features(fn, 22050)
                    ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
                    features = np.vstack([features,ext_features])
                    print(fn.split('/')[2].split('-')[1])
                    if fn.split('/')[2].split('-')[1] == '2':
                        labels = np.append(labels, 1)
                        print("Hello")
                    else:
                        labels = np.append(labels, 0)
                        print("Hi")  
                except Exception as e:
                    print(e)
    return np.array(features), np.array(labels, dtype = np.int)

def encode_matrix(labels, num_unique_labels):
    num_labels = len(labels)
    encoded_matrix = np.zeros((num_labels, num_unique_labels), dtype=int)
    encoded_matrix[np.arange(num_labels), labels] = 1
    return encoded_matrix

if __name__ == "__main__":
    training_steps = int(sys.argv[2])
    num_input = 193
    num_hidden_one = int(sys.argv[3])
    num_hidden_two = int(sys.argv[4])
    learning_rate = 0.01
    num_classes = 2
    percent_split = 70
    file = open("stats.txt", "a+")

    #test_x, test_y = parse_audio_file(os.getcwd() + "/" + sys.argv[1])
    test_x, test_y = parse_audio_files("SoundfilesB", ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], percent_split)
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

    loss = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(prediction), reduction_indices=[1]))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    y_true, y_pred = None, None
    saver = tf.train.Saver()
    with tf.Session() as sess:
        
        #for i in range(0,100):
        saver.restore(sess, "Models/ChildrenPlaying_DogBarking_training_{}_{}B_{}_{}_01x.ckpt".format(percent_split, training_steps, num_hidden_one, num_hidden_two))

        y_pred = sess.run(tf.argmax(prediction,1),feed_dict={X: test_x})
        y_true = sess.run(tf.argmax(test_y,1))
        print("Test accuracy: ",round(sess.run(accuracy, feed_dict={X: test_x,Y: test_y}),3))

    p,r,f,s = precision_recall_fscore_support(y_true, y_pred, average='micro')
    file.write("{} {} {} {}: {} {} {} {} \n".format(percent_split, training_steps, num_hidden_one, num_hidden_two, p, r, f, s))
    print(classification_report(y_true, y_pred))

