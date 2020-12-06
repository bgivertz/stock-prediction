import tensorflow as tf
from model import *
from preprocess import *
import numpy as np
from model.utils import *

def test(model, test_data):
    accuracies = []
    for stock in test_data:
        inputs, labels = generate_input(stock, model.window_size)
        
        num_batches = tf.shape(inputs)[0] // model.batch_size
        for i in range(num_batches):
            batch_input, batch_labels = get_batch(inputs, labels, model.batch_size, model.batch_size*i)

            predictions = model.call(batch_input)
            loss = model.loss(predictions, batch_labels)
            accuracies.append(accuracy(tf.cast(batch_input, tf.float32), tf.cast(predictions, tf.float32), tf.cast(batch_labels, tf.float32), model))
    return np.mean(accuracies)

def accuracy(inputs, predictions, labels, model):
    correct_predictions = 0
    predictions = tf.reshape(predictions, (model.batch_size * model.window_size, -1))
    labels = tf.reshape(labels, (model.batch_size * model.window_size, -1))
    inputs = tf.reshape(inputs, (model.batch_size * model.window_size, -1))
    for idx in range(0, len(predictions)):
        # check if correctly predicted would go up/down
        if np.sign(predictions[idx][0] - inputs[idx][0]) == np.sign(labels[idx][0] - inputs[idx][0]):
            correct_predictions += 1
    return correct_predictions / len(predictions)
