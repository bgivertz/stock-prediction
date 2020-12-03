from preprocess import config
import preprocessor as p
from textblob import TextBlob
import snscrape.modules.twitter as sntwitter
from langdetect import detect
import os
import csv
import numpy as np
from datetime import date, timedelta
import time

'''
generates a dictionary where each key is a hashtag we are querying for,
and each value is a list of tweets between the specified dates, that contain
that hashtag
'''


def scrape_for_tweets(path, date_list):
    for day in date_list:
        start_day = convert_to_date_object(day)
        end_day = start_day + timedelta(days=1)

        per_day_sentiment_list = [day]
        # for each keyword assemble a list of all tweets
        for keyword_idx, keyword in enumerate(config.keyword_list):
            per_day_per_keyword_sentiment_list = []
            # get all tweets on a day with a keyword
            for i, tweet in enumerate(sntwitter.TwitterSearchScraper(
                    f'{keyword} + since:{str(start_day)} until:{str(end_day)}').get_items()):
                try:
                    lan = detect(tweet.content)
                except:
                    lan = 'error'
                if i == config.max_num_tweets:
                    break
                if lan == 'en':
                    p.set_options(p.OPT.URL, p.OPT.EMOJI)
                    clean_tweet = p.clean(tweet.content)
                    # get tweet sentiment
                    tweet_sentiment = TextBlob(clean_tweet).sentiment[0]
                    # list of sentiments for a day for a keyword
                    per_day_per_keyword_sentiment_list.append(tweet_sentiment)

            # average across keyword to get average sentiment on a day for a keyword
            # append this value to list of (1 x keyword) of sentiments for given day
            per_day_sentiment_list.append(avg(per_day_per_keyword_sentiment_list))

        # write each day to csv continuously
        print('\t\twriting day ' + str(day) + ' to csv')
        write_to_csv(path, np.array(per_day_sentiment_list))


def avg(list):
    return sum(list) / len(list)


def write_to_csv(path, arr):
    with open(path, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(arr)


def convert_to_date_object(date_string):
    date_format = date_string.split('-')
    # convert day to date format, and get day after current day
    return date(int(date_format[0]), int(date_format[1]), int(date_format[2]))


def generate_list_of_days_from_stock(path):
    # if stock data exists
    stock_path = path + '/data_' + config.n_days
    stock_file = stock_path + '/' + config.stock_listing_symbols[0] + '-' + config.n_days + '.csv'
    date_list = []
    if os.path.exists(stock_file):
        with open(stock_file, 'r') as current_file:
            csv_reader = csv.reader(current_file, delimiter=',')
            next(csv_reader)
            for line in csv_reader:
                # gets date from stock csv
                date_list.append(line[1])
    else:
        print('no stock data found under the path' + stock_file)
    return date_list


def check_csv_tweet_progress(path):
    with open(path, 'r') as csvfile:
        last_line = csvfile.readlines()[-1].split(',')
        last_date = last_line[0]
        return last_date


def generate_tweet_sentiment_csvs(path):
    stock_dates = generate_list_of_days_from_stock(path)

    # all parameters necessary to extract tweets. used to record if already have csv with tweets extracted using given parameters
    # if do, don't re-scrape
    tweet_extract_params = [config.date_range_start, config.date_range_end,
                            ' '.join([str(elem) for elem in config.keyword_list]), str(config.max_num_tweets)]

    sentiments_file = ''
    found_sentiments_file = False

    # if csv file with sentiments already exists
    sentiments_data_path = path + '/tweet_data/'
    if os.path.exists(sentiments_data_path):
        for file in os.listdir(sentiments_data_path):
            if file.startswith('sentiments-') and os.stat(sentiments_data_path + '/' + file).st_size > 1:
                sentiments_file = sentiments_data_path + '/' + file
                with open(sentiments_file, 'r') as current_file:
                    reader = csv.reader(current_file)
                    header = next(reader)
                    # if csv file has the write parameters don't rescrape
                    if header == tweet_extract_params:
                        print('found file with sentiments already generated using given parameters')
                        csv_tweet_progress = check_csv_tweet_progress(sentiments_file)
                        last_date_converted = convert_to_date_object(csv_tweet_progress)
                        if last_date_converted < convert_to_date_object(stock_dates[-1]):
                            new_start_index = stock_dates.index(last_date_converted.strftime("%Y-%m-%d"))
                            stock_dates = stock_dates[new_start_index + 1: -1]
                            found_sentiments_file = True
                            break

                        else:
                            numpy_sentiments = np.genfromtxt(sentiments_file, delimiter=',',
                                                             skip_header=1)
                            return numpy_sentiments

    # no csv files exist with given parameters, or file exists, but we need to
    # add to it
    # must first make directory if it doesn't exist, and either create new,
    # unique, sentiments file, or use old one
    if not os.path.exists(path + "/tweet_data/"):
        os.makedirs(path + "/tweet_data/")
    if not found_sentiments_file:
        sentiments_file = path + "/tweet_data/sentiments-" + time.strftime("%Y%m%d-%H%M%S") + '.csv'

        with open(sentiments_file, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(tweet_extract_params)

    # generates a 2d array of dimensions number of days x number of keywords, where each index contains average sentiment for that day and keyword
    print(
        '\n\tretrieving tweets with the following hashtag(s) posted btwn ' + stock_dates[0] + " and " + stock_dates[
            -1] + " ...")
    scrape_for_tweets(sentiments_file, stock_dates)
    return sentiments_file


def csv_to_vector(path, csv_path=None):

    if csv_path == None:
        if not os.path.exists(path + "/tweet_data/") or os.listdir(path + "/tweet_data/") == 0:
            print('empty or non-existing tweet data directory')
            return
        files = os.listdir(path + "/tweet_data/")

        # get oldest generated file
        sorted_files = sorted(files, key=lambda timestamp: time.strptime(timestamp[len('sentiments')+1:-4], "%Y%m%d-%H%M%S"))
        csv_path = path + '/tweet_data/' + sorted_files[0]

    tweet_sentiment_vector = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
    tweet_sentiment_vector = tweet_sentiment_vector[:, 1:] #skips first column (date)
    return tweet_sentiment_vector

