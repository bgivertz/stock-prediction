import os   
import argparse
import numpy as np
import preprocess.stock_preprocess as stock_preprocess
import preprocess.tweets_preprocess as tweets_preprocess

from sklearn.preprocessing import MinMaxScaler

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--data", required=True,
                        help="Specify the path to your data file (contains csvs with stock data)")

    parser.add_argument("-v", "--verbose", required=False,
                        help="print verbose output", action='store_true')
                        
    parser.add_argument("-s", "--stocks", required=False,
                        help="only generate stocks", action='store_true')

    args = parser.parse_args()

    verbose = args.verbose

    path = args.data
    if not os.path.isdir(path):
        if os.path.exists(path):
            print("ERROR: path must be a directory")
            exit()
        else:
            print("Creating new dir at path")
            os.makedirs(path)

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

if __name__ == '__main__':
    main()
