import sys
import numpy as np

import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support

file_name = sys.argv[1]
x_data = np.load(file_name)

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('Models/Folder_Training.ckpt.meta')
    saver.restore(sess,tf.train.latest_checkpoint('Models/'))

    graph = tf.get_default_graph()

    x_input = graph.get_tensor_by_name("X:0")
    result = graph.get_tensor_by_name("Result:0")

    #predictions = result.eval(feed_dict={x_input: x_data,})

    y_pred = sess.run(tf.argmax(result,1),feed_dict={x_input: x_data})

print(y_pred)