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

def parse_audio_files(file_name,file_ext='*.wav'):
    features = np.empty((0,193))
    try:
        mfccs, chroma, mel, contrast,tonnetz = extract_feature(file_name)
        ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
        features = np.vstack([features,ext_features])
    except Exception as e:
        print(e)
    return np.array(features)


file_name = "Soundfiles/TestSounds/Dog_Barking_Test1.wav"
x_data = parse_audio_files(file_name)

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('Models/Folder_Training.ckpt.meta')
    saver.restore(sess,tf.train.latest_checkpoint('Models/'))

    graph = tf.get_default_graph()

    x_input = graph.get_tensor_by_name("X:0")
    result = graph.get_tensor_by_name("Result:0")

    #predictions = result.eval(feed_dict={x_input: x_data,})

    y_pred = sess.run(tf.argmax(result,1),feed_dict={x_input: x_data})

print(y_pred)