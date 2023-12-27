import glob
import os
import librosa
import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

def extract_feature(file_name):
    x, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
    mfccs = np.mean(librosa.feature.mfcc(x, sr=sample_rate, n_mfcc=40).T,axis=0)
    return mfccs

def parse_audio_files(parent_dir,sub_dirs,file_ext='*.wav'):
    features = []
    labels = []
    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            try:
                features.append(extract_feature(fn))
                labels.append(fn.split('/')[2].split('-')[1])
            except Exception as e:
                print(e)
    return np.array(features), labels

parent_dir = 'Soundfiles'

sub_dirs = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5']
x, labels = parse_audio_files(parent_dir,sub_dirs)

label_encoder = LabelEncoder()
y = tf.keras.utils.to_categorical(label_encoder.fit_transform(labels))

train_test_split = np.random.rand(len(x)) < 0.70
train_x = x[train_test_split]
train_y = y[train_test_split]
test_x = x[~train_test_split]
test_y = y[~train_test_split]

print('Shape of Features(Train): ', x.shape)
print('Shape of Labels(Train): ', y.shape)

training_epochs = 100
model=tf.keras.models.Sequential([
    tf.keras.layers.Dense(128,activation='relu',input_shape=(40,)),
    
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(256,activation='relu'),
    tf.keras.layers.Dense(256,activation='relu'),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dropout(0.60),

    tf.keras.layers.Dense(10,activation='softmax')
])
     

reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, mode='auto')
model.compile(optimizer='sgd',loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_x, train_y, batch_size=32, epochs=training_epochs, validation_data=(test_x, test_y), callbacks=[reduce])

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs=range(len(acc)) #No. of epochs

#Plot training and validation accuracy per epoch
plt.figure(0)
plt.plot(epochs,acc,'b',label='Training Accuracy')
plt.plot(epochs,val_acc,'r',label='Validation Accuracy')
plt.legend()
plt.grid(which='major', linestyle = '-', linewidth = 0.5)
plt.minorticks_on()
plt.grid(which='minor', linestyle='--', linewidth = 0.5)
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.savefig('accuracy.png')
plt.close()

#Plot training and validation loss per epoch
plt.figure(1)
plt.plot(epochs,loss,'b',label='Training Loss')
plt.plot(epochs,val_loss,'r',label='Validation Loss')
plt.legend()
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.savefig('loss.png')