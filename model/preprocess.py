def get_batch(train_inputs, train_labels, batch_size, start_index):
	
	batch_inputs = train_inputs[start_index : start_index + batch_size, :, :]
	batch_labels = train_labels[start_index : start_index + batch_size, :, :]

	return batch_inputs, batch_labels