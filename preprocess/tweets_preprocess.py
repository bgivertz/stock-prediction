from preprocess import config
import preprocessor as p
from textblob import TextBlob
import snscrape.modules.twitter as sntwitter
from langdetect import detect

'''
generates a dictionary where each key is a hashtag we are querying for,
and each value is a list of tweets between the specified dates, that contain
that hashtag
'''
def scrape_for_tweets():
    tweets = {}
    for keyword in config.keyword_list:
        tweets[keyword] = []
        for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f'{keyword} + since:{config.date_range_start} until:{config.date_range_end}').get_items()):
            try:
                lan = detect(tweet.content)
            except:
                lan = 'error'
            if i == config.max_num_tweets:
                break
            if lan == 'en':
                tweets[keyword].append(tweet.content)
    return tweets

'''
:param tweets: dictionary where each key is a hashtag we are querying for,
and each value is a list of tweets between the specified dates, that contain
that hashtag

:returns same tweets dictionary, but at each keyword, the list of tweets are cleaned
'''
def clean_tweets(tweets):
    for keyword in tweets:
        p.set_options(p.OPT.URL, p.OPT.EMOJI)
        tweets[keyword] = [p.clean(tweet) for tweet in tweets[keyword]]
    return tweets

def avg(list):
    return sum(list)/len(list)

def generate_tweet_sentiments():
    # generates a dictionary  where each key is a hashtag we are querying for,
    # and each value is a list of tweets between the specified dates, that contain
    # the hashtag in config.keyword_list
    print('\n\tretrieving tweets with the hashtag(s): [#' + ', #'.join(
        config.keyword_list) + "] posted btwn " + config.date_range_start + " and " + config.date_range_end + " ...")
    tweets = clean_tweets(scrape_for_tweets())
    #returns list of tuples (polarity, subjectivity) polarity is between -1 (negative statement) and 1 (postiive statement)
    # and subjectivity is betweenn 0 (very objective, non-personal, statement) and 1 (very subjective, personal, statement)
    print('\n\tretrieved information for ' + str(sum(len(v) for v in tweets.values())) + ' tweets')
    #a dictionary of each keyword mapped to a list of the sentiment for each tweet containing the keyword
    tweet_sentiments = {}
    #a dictionary containing each keyword mapped to the average sentiment for all posts containing that keyword
    average_tweet_sentiments = {}
    for keyword in tweets:
        tweet_sentiments[keyword] = [TextBlob(tweet).sentiment for tweet in tweets[keyword]]
        average_tweet_sentiments[keyword] = list(map(avg, zip(*tweet_sentiments[keyword])))
    return tweet_sentiments

