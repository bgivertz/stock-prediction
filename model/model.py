import tensorflow as tf
from tensorflow.keras import Model
import numpy as np

class StockModel(tf.keras.Model):
    def __init__(self):

        super(StockModel, self).__init__()

        self.output_size = 2
        self.window_size = 5 #Need to update
        self.input_size = 32
        self.batch_size = 5
        self.learning_rate = 0.1

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.LSTM = tf.keras.layers.LSTM(128, return_sequences=True, return_state=True)
        self.D1 = tf.keras.layers.Dense(units=500, activation='relu')
        self.D2 = tf.keras.layers.Dense(units=self.output_size, activation='softmax')


    def call(self, inputs):

        #May switch to LSTM
        whole_seq_output, _, _ = self.LSTM(inputs=inputs)

        x = self.D1(whole_seq_output)
        probs = self.D2(x)
        
        return probs

    def loss(self, probs, labels):
        """returns average batch loss"""

        loss = tf.keras.losses.MSE(labels, probs)
        loss = tf.reduce_mean(loss)

        return loss