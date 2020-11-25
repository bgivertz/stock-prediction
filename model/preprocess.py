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