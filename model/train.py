import tensorflow as tf
from model import *
from preprocess import *
import numpy as np
from model.utils import *

def train(model, train_data):
    """
    This method should train the model for a single company and then be called again for the another 
    company. This means this model will be called multiple times per epoch. 
    train_inputs: [num_inputs, length_input(aka stock + twitter)] (should be floats)
    in the future. (should be floats)
    """
    for epoch in range(model.epochs):
        losses = []
        print(f'EPOCH {epoch + 1}/{model.epochs}')
        for stock in train_data:
            inputs, labels = generate_input(stock, model.window_size)
            
            inds = tf.range(inputs.shape[0])
            tf.random.shuffle(inds)
            inputs = tf.gather(inputs, inds)
            labels = tf.gather(labels, inds)

            num_batches = tf.shape(inputs)[0] // model.batch_size
            for i in range(num_batches):
                batch_input, batch_labels = get_batch(inputs, labels, model.batch_size, model.batch_size*i)
                with tf.GradientTape() as tape:
                    predictions = model.call(batch_input)
                    loss = model.loss(predictions, batch_labels)
                    losses.append(loss)
                
                gradients = tape.gradient(loss, model.trainable_variables)
                model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        print(f'Average loss: {np.mean(losses)}')