import tensorflow as tf
from model import model, preprocess, train
from preprocess import *
import numpy as np



def train(model, train_inputs, train_labels):
    """
    This method should train the model for a single company and then be called again for the another 
    company. This means this model will be called multiple times per epoch. 
    train_inputs: [num_inputs, length_input(aka stock + twitter)] (should be floats)
    train_label: [num_inputs, model.output_size(open and close and maybe others)] This should be from one day
    in the future. (should be floats)
    """

    #number of inputs divided by the window size to see how many windows we can get
    num_winds = tf.shape(train_inputs)[0] // model.window_size
    
    # assert num_winds == 200

    train_inputs = tf.reshape(train_inputs[0:num_winds * model.window_size, :], (num_winds, model.window_size, model.input_size))

    # assert train_inputs.shape[0] == 200
    # assert train_inputs.shape[1] == 5
    # assert train_inputs.shape[2] == 20

    all_loss = []

    num_batches = (num_winds * model.window_size) // model.batch_size

    for i in range(num_batches):
        batch_input, batch_label = preprocess.get_batch(train_inputs, train_labels, model.batch_size, model.batch_size*i)

        with tf.GradientTape() as tape:
            predictions = model.call(batch_input, None)
            loss = model.loss(predictions, batch_label)

        all_loss.append(loss)

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        #keep track of the progress
        if i % 5 == 0:
            print("batch", i, "of", num_batches, " // Avg Loss: ", loss/model.batch_size, " // Perplexity: ", np.exp(np.mean(all_loss)))
    
    #perplexity, not sure if it is helpful
    return np.exp(np.mean(all_loss)) 

def main():

    #get_data()
    
    train_inputs = np.arange(20000, dtype=float)
    train_labels = np.ones((1000,2))

    train_inputs = np.reshape(train_inputs, (1000, 20))

    model = StockModel()

    print("DONE: ", train(model, train_inputs, train_labels))
    


if __name__ == '__main__':
    main()