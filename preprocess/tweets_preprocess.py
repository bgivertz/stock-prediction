from preprocess import config
import preprocessor as p
from textblob import TextBlob
import snscrape.modules.twitter as sntwitter
from langdetect import detect

# generates a txt files with links to all tweets containing certain key words
def scrape_for_tweets():
    tweets = []
    for keyword in config.keyword_list:
        for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f'{keyword} + since:{config.date_range_start} until:{config.date_range_end}').get_items()):
            try:
                lan = detect(tweet.content)
            except:
                lan = 'error'
            if i == config.max_num_tweets:
                break
            if lan == 'en':
                tweets.append(tweet.content)
    return tweets

def clean_tweets(tweets):
    p.set_options(p.OPT.URL, p.OPT.EMOJI)
    return [p.clean(tweet) for tweet in tweets]

def generate_tweet_sentiments():
    # generates a txt files with links to all tweets containing certain key words
    print('\n\tretrieving tweets with the hashtag(s): [#' + ', #'.join(
        config.keyword_list) + "] posted btwn " + config.date_range_start + " and " + config.date_range_end + " ...")
    tweets = clean_tweets(scrape_for_tweets())
    #returns list of tuples (polarity, subjectivity) polarity is between -1 (negative statement) and 1 (postiive statement)
    # and subjectivity is betweenn 0 (very objective, non-personal, statement) and 1 (very subjective, personal, statement)
    print('\n\tretrieved information for ' + str(len(tweets)) + ' tweets')
    tweet_sentiments = [TextBlob(tweet).sentiment for tweet in tweets]
    return tweet_sentiments

