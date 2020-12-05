import tensorflow as tf
from model import *
from preprocess import *
import numpy as np
from model import train


def test(model, test_inputs):
    """
    This method should train the model for a single company and then be called again for the another 
    company. This means this model will be called multiple times per epoch. 
    train_inputs: [num_inputs, length_input(aka stock + twitter)] (should be floats)
    in the future. (should be floats)
    """

    # number of inputs divided by the window size to see how many windows we can get
    num_winds = (tf.shape(test_inputs)[0]) // model.window_size

    test_inputs = test_inputs[0:num_winds * model.window_size]
    test_labels = test_inputs.copy()

    test_inputs = test_inputs[:-(model.window_size)]
    test_labels = test_labels[1:-(model.window_size - 1)]

    print(test_inputs.shape)

    test_inputs = tf.reshape(test_inputs, (-1, model.window_size, model.input_size))

    print(test_inputs.shape)

    test_labels = tf.reshape(test_labels, (-1, model.window_size, model.input_size))
    test_labels = test_labels[:, :, 0:2]

    all_loss = []
    all_accuracy = []

    num_batches = (tf.shape(test_inputs)[0]) // model.batch_size

    print(tf.shape(test_inputs)[0])

    for i in range(num_batches):
        batch_input, batch_label = train.get_batch(test_inputs, test_labels, model.batch_size, model.batch_size * i)

        predictions = model.call(batch_input)
        loss = model.loss(predictions, batch_label)

        all_accuracy.append(accuracy(tf.cast(batch_input, tf.float32), tf.cast(predictions, tf.float32), tf.cast(batch_label, tf.float32), model))
        all_loss.append(loss)

        # keep track of the progress
        print("batch", i, "of", num_batches, " // Avg Loss: ", loss / model.batch_size)

    return np.mean(all_accuracy)


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
