import tensorflow as tf

def generate_input(stock, window_size):
    inputs = stock[:-1]
    labels = stock[1:]

    num_winds = (tf.shape(inputs)[0]) // window_size
    inputs = inputs[:num_winds * window_size]
    labels = labels[:num_winds * window_size]

    inputs = tf.reshape(inputs, (-1, window_size, inputs.shape[-1]))
    labels = tf.reshape(labels, (-1, window_size, inputs.shape[-1]))

    return inputs, labels

def get_batch(train_inputs, train_labels, batch_size, start_index):
	batch_inputs = train_inputs[start_index : start_index + batch_size, :]
	batch_labels = train_labels[start_index : start_index + batch_size, :]
	return batch_inputs, batch_labels