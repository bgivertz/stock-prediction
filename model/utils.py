import tensorflow as tf

def generate_input(stock, window_size):
    num_winds = (tf.shape(stock)[0]) // window_size

    inputs = stock[:num_winds * window_size]
    labels = inputs.copy()

    inputs = inputs[:-(window_size)]
    labels = inputs[1:-(window_size - 1)]

    vector_size = inputs.shape[-1]
    inputs = tf.reshape(inputs, (-1, window_size, vector_size))
    labels = tf.reshape(labels, (-1, window_size, vector_size))

    return inputs, labels

def get_batch(train_inputs, train_labels, batch_size, start_index):
	batch_inputs = train_inputs[start_index : start_index + batch_size, :, :]
	batch_labels = train_labels[start_index : start_index + batch_size, :, :]
	return batch_inputs, batch_labels