import tensorflow as tf
import csv

def get_data(stock_file):
    context = []
    stocks = []
    with open(stock_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        reader.__next__()
        for row in reader:
            context.append(row[:2])
            stocks.append(list(map(float, row[2:])))
    return context, stocks

def get_batch(train_inputs, train_labels, batch_size, start_index):
	
	batch_inputs = train_inputs[start_index : start_index + batch_size, :, :]
	batch_labels = train_labels[start_index : start_index + batch_size, :, :]

	return batch_inputs, batch_labels