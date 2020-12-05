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
        self.LSTM1 = tf.keras.layers.LSTM(50, return_sequences=True, return_state=True, dtype=tf.float32)
        self.dropout1 = tf.keras.layers.Dropout(.2)
        self.LSTM2 = tf.keras.layers.LSTM(50, return_sequences=True, return_state=True, dtype=tf.float32)
        self.dropout2 = tf.keras.layers.Dropout(.2)
        self.LSTM3 = tf.keras.layers.LSTM(50, return_sequences=True, return_state=True, dtype=tf.float32)
        self.dropout3 = tf.keras.layers.Dropout(.2)
        self.D1 = tf.keras.layers.Dense(units=self.output_size, activation='linear', dtype=tf.float32)
        #self.D2 = tf.keras.layers.Dense(units=self.output_size, activation='relu', dtype=tf.float32)


    def call(self, inputs):

        #May switch to LSTM
        lstm1_seq_output, _, _ = self.LSTM1(inputs=inputs)
        dropout1 = self.dropout1(lstm1_seq_output)
        lstm2_seq_output, _, _ = self.LSTM2(dropout1)
        dropout2 = self.dropout2(lstm2_seq_output)
        lstm3_seq_output, _, _ = self.LSTM3(dropout2)
        dropout3 = self.dropout3(lstm3_seq_output)
        probs = self.D1(dropout3)

        # x = self.D1(whole_seq_output)
        # probs = self.D2(x)
        
        return probs

    def loss(self, probs, labels):
        """returns average batch loss"""

        loss = tf.keras.losses.MSE(labels, probs)
        loss = tf.reduce_mean(loss)

        return loss