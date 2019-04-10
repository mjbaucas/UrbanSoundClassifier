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

def parse_audio(parent_dir, sub_dirs, percent_split, file_ext='*.txt'):
    training_dir = "{}/TrainingSet/{}%".format(parent_dir, str(percent_split))
    if not os.path.exists(training_dir):
         os.makedirs(training_dir)
    features, labels = np.empty((0,193)), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            sound_name_split = fn.split('/')[-1].split('.')[0]
            if os.path.isfile("{}/{}.txt".format(training_dir, sound_name_split)):
                print(fn)
                try:
                    mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn, 22050)
                    ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
                    features = np.vstack([features,ext_features])
                    labels = np.append(labels, fn.split('/')[2].split('-')[1])
                except Exception as e:
                    print(e)
    return np.array(features), np.array(labels, dtype = np.int)

def encode_matrix(labels, num_unique_labels):
    num_labels = len(labels)
    encoded_matrix = np.zeros((num_labels, num_unique_labels), dtype=int)
    encoded_matrix[np.arange(num_labels), labels] = 1
    return encoded_matrix

if __name__ == "__main__":
    parent_dir = 'SoundfilesB'
    #train_sub_dirs = ['8000']
    train_sub_dirs = ['3']
    percent_split = 70
    num_classes = 10
    
    train_x, train_y = parse_audio(parent_dir,train_sub_dirs, percent_split)
    train_y = encode_matrix(train_y, num_classes)

    training_steps = 5000
    num_input = train_x.shape[1]
    num_hidden_one = 280
    num_hidden_two = 300
    learning_rate = 0.01
    std_dev = 1 / np.sqrt(num_input)
    
    X = tf.placeholder(tf.float32,[None,num_input])
    Y = tf.placeholder(tf.float32,[None,num_classes])

    weights_h1 = tf.Variable(tf.random_normal([num_input,num_hidden_one], mean = 0, stddev=std_dev))
    weights_h2 = tf.Variable(tf.random_normal([num_hidden_one,num_hidden_two], mean = 0, stddev=std_dev))
    weights_out = tf.Variable(tf.random_normal([num_hidden_two,num_classes], mean = 0, stddev=std_dev))

    bias_b1 = tf.Variable(tf.random_normal([num_hidden_one], mean = 0, stddev=std_dev))
    bias_b2 = tf.Variable(tf.random_normal([num_hidden_two], mean = 0, stddev=std_dev))
    bias_out = tf.Variable(tf.random_normal([num_classes], mean = 0, stddev=std_dev))

    first_layer = tf.nn.tanh(tf.matmul(X,weights_h1) + bias_b1)
    second_layer = tf.nn.sigmoid(tf.matmul(first_layer,weights_h2) + bias_b2)
    out_layer = tf.matmul(second_layer, weights_out) + bias_out
    prediction = tf.nn.softmax(out_layer)

    loss = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(prediction), reduction_indices=[1]))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        for step in range(training_steps):            
           sess.run([optimizer,loss],feed_dict={X:train_x,Y:train_y})
        
        #save_path = saver.save(sess, "Models/{}_training_{}.ckpt".format("_".join(train_sub_dirs), percent_split))
        save_path = saver.save(sess, "Models/{}_training_{}_2000.ckpt".format("Full", percent_split))