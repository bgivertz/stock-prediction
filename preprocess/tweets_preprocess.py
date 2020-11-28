from preprocess import config
import preprocessor as p
from textblob import TextBlob
import snscrape.modules.twitter as sntwitter
from langdetect import detect
import os
import csv
import numpy as np
from datetime import date, timedelta
'''
generates a dictionary where each key is a hashtag we are querying for,
and each value is a list of tweets between the specified dates, that contain
that hashtag
'''
def scrape_for_tweets():
    start_date = config.date_range_start.split('-')
    end_date = config.date_range_end.split('-')
    formatted_start_date =  date(int(start_date[0]), int(start_date[1]), int(start_date[2]))
    formatted_end_date = date(int(end_date[0]), int(end_date[1]), int(end_date[2]))
    num_days = formatted_end_date - formatted_start_date #year-month-day (0000-00-00)
    tweets_over_all_days = []

    for day in range(num_days.days):
        print('\t\t\t day ' + str(day))
        begin_singular_day = formatted_start_date + timedelta(day)
        end_singular_day = formatted_start_date + timedelta(day+1)
        tweets_on_singular_day = []
        for keyword_idx, keyword in enumerate(config.keyword_list):
            print('\t\t\t\t #' + keyword)
            tweets_on_singular_day_with_keyword = []
            for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f'{keyword} + since:{begin_singular_day} until:{end_singular_day}').get_items()):
                try:
                    lan = detect(tweet.content)
                except:
                    lan = 'error'
                if i == config.max_num_tweets:
                    break
                if lan == 'en':
                    p.set_options(p.OPT.URL, p.OPT.EMOJI)
                    tweets_on_singular_day_with_keyword.append(p.clean(tweet.content))
            tweets_on_singular_day.append(tweets_on_singular_day_with_keyword)
        tweets_over_all_days.append(tweets_on_singular_day)
    return tweets_over_all_days

def avg(list):
    return sum(list)/len(list)

def generate_tweet_sentiments(path):
    # all parameters necessary to extract tweets. used to record if already have csv with tweets extracted using given parameters
    # if do, don't re-scrape
    tweet_extract_params = [config.date_range_start, config.date_range_end, ' '.join([str(elem) for elem in config.keyword_list]) , str(config.max_num_tweets)]

    #if csv file with sentiments already exists, don't rescrape
    sentiments_data_path = path + '/tweet_data/sentiments.csv'
    if os.path.exists(sentiments_data_path):
        with open(sentiments_data_path, 'r') as current_file:
            print('found file with sentiments already generated using given parameters')
            reader = csv.reader(current_file)
            header = next(reader)
            if header == tweet_extract_params:
                numpy_sentiments = np.genfromtxt(path + '/tweet_data/sentiments.csv', delimiter=',', skip_header=1)
                return numpy_sentiments


    # generates a 3d array of dimensions number of days x number of keywords x number of tweets per keyword
    print('\n\tretrieving tweets with the following hashtag(s) posted btwn ' + config.date_range_start + " and " + config.date_range_end + " ...")
    tweets = scrape_for_tweets()

    #creating 2d array of size number of days X number of keywords where each index contains average sentiment for that day and keyword
    total_tweet_sentiment = []
    for day in tweets:
        keyword_sentiments_for_day = []
        for keyword_idx, keyword in enumerate(config.keyword_list):
            tweet_sentiments = [TextBlob(tweet).sentiment for tweet in day[keyword_idx]]
            keyword_sentiments_for_day.append(list(map(avg, zip(*tweet_sentiments)))[0])
        total_tweet_sentiment.append(keyword_sentiments_for_day)

    # write sentiments to csv file so don't have to regenerate every time
    numpy_sentiments = np.array(total_tweet_sentiment)
    if not os.path.exists(path + "/tweet_data/"):
        os.makedirs(path + "/tweet_data/")
    with open(path + "/tweet_data/sentiments.csv", 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(tweet_extract_params)
        csvwriter.writerows(numpy_sentiments.tolist())

    return numpy_sentiments

