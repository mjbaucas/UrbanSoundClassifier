import glob
import os
import librosa
import numpy as np

import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support

def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz

def parse_audio_files(parent_dir,sub_dirs,file_ext='*.wav'):
    features, labels = np.empty((0,193)), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            try:
                mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn)
                ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
                features = np.vstack([features,ext_features])
                labels = np.append(labels, fn.split('/')[2].split('-')[1])
            except Exception as e:
                print(e)
    return np.array(features), np.array(labels, dtype = np.int)

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode

parent_dir = 'Soundfiles'

sub_dirs = ['fold1']
x_data, labels = parse_audio_files(parent_dir,sub_dirs)

y_data = one_hot_encode(labels)

y_true, y_pred = None, None
with tf.compat.v1.Session() as sess:
    saver = tf.compat.v1.train.import_meta_graph('Models/Folder_Training.ckpt.meta')
    saver.restore(sess,tf.compat.v1.train.latest_checkpoint('Models/'))

    graph = tf.compat.v1.get_default_graph()

    x_input = graph.get_tensor_by_name("X:0")
    result = graph.get_tensor_by_name("Result:0")
    
    y_pred = sess.run(tf.argmax(result,1),feed_dict={x_input: x_data})
    y_true = sess.run(tf.argmax(y_data,1))

p,r,f,s = precision_recall_fscore_support(y_true, y_pred, average='micro')
print("F-Score: {}".format(round(f,3)))