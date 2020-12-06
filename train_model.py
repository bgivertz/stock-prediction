import os
from preprocess import stock_preprocess, tweets_preprocess
import argparse
import numpy as np
from model.model import StockModel
from model.train import train
from model.test_loop import test

from sklearn.preprocessing import MinMaxScaler

def create_data(stock_data, twitter_data, no_sentiment):
    test_ratio = 0.1

    train_data = []
    test_data = []
    for stock in stock_data:
        stock = stock[:len(twitter_data)][:]
        if not no_sentiment:
            stock = np.concatenate((stock, twitter_data), axis=1)
    
        divider = len(stock) - int(len(stock) * test_ratio)
        train_data.append(stock[:divider])
        test_data.append(stock[divider:])
    return train_data, test_data

def get_stock_data(stock_files):
    stock_vector_list = []
    for stock_file in stock_files:
        stock_vector_list.append(stock_preprocess.csv_to_vector(stock_file))

    # Normalize
    scaler = MinMaxScaler(feature_range=(0, 1))
    for stock in range(len(stock_vector_list)):
        stock_vector_list[stock] = scaler.fit_transform(stock_vector_list[stock])

    return stock_vector_list

def get_twitter_data(twitter_file):
    twitter_data = tweets_preprocess.csv_to_vector(None, twitter_file)

    # Normalize
    scaler = MinMaxScaler(feature_range=(0, 1))
    twitter_data = scaler.fit_transform(twitter_data)
    return twitter_data


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-sd", "--stock_data", required=True,
                        help="Specify the path to your stock data")
    parser.add_argument("-td", "--twitter_data", required=True,
                        help="Specify the path to your Twitter data")
    parser.add_argument("-v", "--verbose", required=False,
                        help="print verbose output", action='store_true')
    parser.add_argument("-ns", "--no_sentiment", required=False,
                        help="train without sentiment", action='store_true')
    
    args = parser.parse_args()

    stock_path = args.stock_data
    twitter_path = args.twitter_data
    verbose = args.verbose
    no_sentiment = args.no_sentiment

    stock_files = stock_preprocess.get_stock_csv_files(stock_path)
    twitter_file = os.path.join(twitter_path, os.listdir(twitter_path)[0])

    stock_data = get_stock_data(stock_files)
    twitter_data = get_twitter_data(twitter_file)

    train_data, test_data = create_data(stock_data, twitter_data, no_sentiment)

    model = StockModel(14 if no_sentiment else 32)
    train(model, train_data)
    print(test(model, test_data))


if __name__ == '__main__':
    main()
