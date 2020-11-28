import os
from preprocess import stock_preprocess, tweets_preprocess
import argparse


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--data", required=True,
                        help="Specify the path to your data file (contains csvs with stock data)")

    parser.add_argument("-v", "--verbose", required=False,
                        help="print verbose output")

    args = parser.parse_args()

    if os.path.isdir(args.data):
        path = args.data
    else:
        print("ERROR: path must be a directory")
        exit()

    verbose = True
    if args.verbose == None:
        verbose = False


    '''
    STOCK PREPROCESS
      uses csv files from yahoo finance of stock information to create several directories of parsed csv files
      that contain stock data over different denominations of past time periods'''
    print('running stock preprocess... ')
    stock_preprocess.generate_stock_csvs(path, verbose)

    '''
    TWITTER PREPROCESS
    scrapes twitter for tweets containing hashtags (specified in config.py) over a certain time period
    and runs sentiment analysis on them, returns a dictionary containing hashtag mapped to
    [average sentiment, average polarity] for all posts containing that hashtag'''
    print('running tweet preprocess... ')
    twitter_sentiments = tweets_preprocess.generate_tweet_sentiments()



if __name__ == '__main__':
    main()