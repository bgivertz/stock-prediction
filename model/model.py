import tensorflow as tf
from tensorflow.keras import Model
import numpy as np

class StockModel(tf.keras.Model):
    def __init__(self):

        super(StockModel, self).__init__()

        self.output_size = 2
        self.window_size = 5 #Need to update
        self.input_size = 20
        self.batch_size = 100
        self.learning_rate = 0.01

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.GRU = tf.keras.layers.GRU(128, return_sequences=True, return_state=True)
        self.D1 = tf.keras.layers.Dense(units=500, activation='relu')
        self.D2 = tf.keras.layers.Dense(units=self.output_size, activation='softmax')


    def call(self, inputs, initial_state):

        print(inputs)
        #Is GRU okay? Should we switch to LSTM
        whole_seq_output, final_state = self.GRU(inputs=inputs, initial_state=initial_state)

        x = self.D1(whole_seq_output)
        probs = self.D2(x)
        
        return probs

    def loss(self, probs, labels):

        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, probs)

        return tf.math.reduce_mean(loss)



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
    
    assert num_winds == 200

    train_inputs = tf.reshape(train_inputs[0:num_winds * model.window_size], (num_winds, model.window_size, model.input_size))

    assert train_inputs.shape[0] == 200
    assert train_inputs.shape[1] == 5
    assert train_inputs.shape[2] == 20

    all_loss = []

    num_batches = (num_winds * model.window_size) // model.batch_size

    for i in range(num_batches):
        batch_input, batch_label = get_batch(train_inputs, train_labels, model.batch_size, model.batch_size*i)

        with tf.GradientTape() as tape:
            predictions = model.call(batch_input, None)
            loss = model.loss(predictions, batch_label)

        all_loss.append(loss)

        # gradients = tape.gradient(loss, model.trainable_variables)
        # model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        #keep track of the progress
        if i % 5 == 0:
            print("batch", i, "of", num_batches, " // Avg Loss: ", loss/model.batch_size, " // Perplexity: ", np.exp(np.mean(all_loss)))
    
    #perplexity, not sure if it is helpful
    return np.exp(np.mean(all_loss)) 

def get_batch(train_inputs, train_labels, batch_size, start_index):
	
	batch_inputs = train_inputs[start_index : start_index + batch_size, :, :]
	batch_labels = train_labels[start_index : start_index + batch_size, :]

	return batch_inputs, batch_labels

def main():
    train_inputs = np.arange(20000, dtype=float)
    train_labels = np.ones((1000,2))

    train_inputs = np.reshape(train_inputs, (1000, 20))

    model = StockModel()

    print("DONE: ", train(model, train_inputs, train_labels))

if __name__ == '__main__':
    main()