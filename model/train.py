import tensorflow as tf
from model import *
from preprocess import *
import numpy as np


def get_batch(train_inputs, train_labels, batch_size, start_index):

	batch_inputs = train_inputs[start_index : start_index + batch_size, :, :]
	batch_labels = train_labels[start_index : start_index + batch_size, :, :]

	return batch_inputs, batch_labels


def train(model, train_inputs):
    """
    This method should train the model for a single company and then be called again for the another 
    company. This means this model will be called multiple times per epoch. 
    train_inputs: [num_inputs, length_input(aka stock + twitter)] (should be floats)
    in the future. (should be floats)
    """

    #number of inputs divided by the window size to see how many windows we can get
    num_winds = (tf.shape(train_inputs)[0]) // model.window_size
    
    train_inputs = train_inputs[0:num_winds * model.window_size]
    train_labels = train_inputs.copy()

    train_inputs = train_inputs[:-(model.window_size)]
    train_labels = train_labels[1:-(model.window_size - 1)]

    print(train_inputs.shape)

    train_inputs = tf.reshape(train_inputs, (-1, model.window_size, model.input_size))

    print(train_inputs.shape)

    train_labels = tf.reshape(train_labels, (-1, model.window_size, model.input_size))
    train_labels = train_labels[:, :, 0:2]

    all_loss = []

    num_batches = (tf.shape(train_inputs)[0]) // model.batch_size

    print(tf.shape(train_inputs)[0])

    for i in range(num_batches):
        batch_input, batch_label = get_batch(train_inputs, train_labels, model.batch_size, model.batch_size*i)

        with tf.GradientTape() as tape:
            predictions = model.call(batch_input)
            loss = model.loss(predictions, batch_label)

        all_loss.append(loss)

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        #keep track of the progress
        print("batch", i, "of", num_batches, " // Avg Loss: ", loss/model.batch_size)
    
    #perplexity, not sure if it is helpful
    return np.exp(np.mean(all_loss)) 

def main():

    #get_data()
    
    train_inputs = np.arange(60000, dtype=float)

    train_inputs = np.reshape(train_inputs, (-1, 20))

    model = StockModel()

    print("DONE: ", train(model, train_inputs))
    


if __name__ == '__main__':
    main()