import os
from preprocess import stock_preprocess, tweets_preprocess
import argparse
import numpy as np
from model.model import StockModel
from model.train import train
from model.test_loop import test

from sklearn.preprocessing import MinMaxScaler


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--data", required=True,
                        help="Specify the path to your data file (contains csvs with stock data)")

    parser.add_argument("-v", "--verbose", required=False,
                        help="print verbose output", action='store_true')

    parser.add_argument("-s", "--stocks", required=False,
                        help="only print stocks", action='store_true')

    parser.add_argument("-ns", "--no_sentiment", required=False,
                        help="don't train with sentiment", action='store_true')

    args = parser.parse_args()

    if os.path.isdir(args.data):
        path = args.data
    else:
        print("ERROR: path must be a directory")
        exit()

    verbose = args.verbose

    '''
    STOCK PREPROCESS
      uses csv files from yahoo finance of stock information to create several directories of parsed csv files
      that contain stock data over different denominations of past time periods'''
    print('running stock preprocess... ')
    stock_files = stock_preprocess.generate_stock_csvs(path, verbose)

    stock_vector_list = []
    for file in stock_files:
        stock_vector_list.append(stock_preprocess.csv_to_vector(file))

    '''
    TWITTER PREPROCESS
     scrapes twitter for tweets containing hashtags (specified in config.py) over a certain time period
    and runs sentiment analysis on them, outputs into csv'''
    if not args.stocks:
        print('running tweet preprocess... ')
        tweets_csv = tweets_preprocess.generate_tweet_sentiment_csvs(path)
        tweets_vector = tweets_preprocess.csv_to_vector(path, tweets_csv)

    else:
        print('getting tweet data from most recently create tweet file... ')
        print('')
        tweets_vector = tweets_preprocess.csv_to_vector(path)

    model = StockModel()

    #normalize

    scaler = MinMaxScaler(feature_range=(0, 1))
    tweets_vector = scaler.fit_transform(tweets_vector)

    num_test_values = 500
    tweets_vector_train = tweets_vector[:-num_test_values][:]
    tweets_vector_test = tweets_vector[-num_test_values:][:]

    for stock in stock_vector_list:
        print('\n\n NEW STOCK')
        stock = scaler.fit_transform(stock)
        abbreviated_stock_vector = stock[:len(tweets_vector_train)][:]
        if args.no_sentiment:
            train_inputs = stock
            model.input_size = 14

        else:
            train_inputs = np.concatenate((abbreviated_stock_vector, tweets_vector_train), axis=1)
            model.input_size = 32

        train(model, train_inputs)


    print('\n\n\nTESTING')
    for stock in stock_vector_list:
        print('\n\n NEW STOCK')
        stock = scaler.fit_transform(stock)
        abbreviated_stock_vector_test = stock[-len(tweets_vector_test):][:]
        if args.no_sentiment:
            test_inputs = stock
            model.input_size = 14

        else:
            test_inputs = np.concatenate((abbreviated_stock_vector_test, tweets_vector_test), axis=1)
            model.input_size = 32
        print(test(model, test_inputs))


if __name__ == '__main__':
    main()
