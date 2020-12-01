import tensorflow as tf
from model import *
from preprocess import *

def train(model, context, stocks):
    pass

def test(model, context, stocks):
    pass

def get_data_from_path(path):
    return [],[],[],[]

def main():
    model = StockModel()

    train_context, train_stocks, test_context, test_stocks = get_data_from_path('data/data_10')

    for c, s in zip(train_context, train_stocks):
        train(model, c, s)
    
    acc = []
    for c, s in zip(test_context, test_stocks):
        acc.append(test(model, c, s))
    print(tf.reduce_mean(acc))
    


if __name__ == '__main__':
    main()