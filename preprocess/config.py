# TWITTER PREPROCESS QUERY PARAMETERS
# format: year-month-day (0000-00-00)
date_range_start = '2008-01-01'
date_range_end = '2020-11-01'
max_num_tweets = 100
#removed avgo because doesn't have data before 2009

stock_listing_symbols = ['AAPL', 'ACN', 'ADBE', 'ADBE', 'AMD', 'CAJ', 'CSCO', 'GOOGL', 'GRMN', 'IBM', 'INTC', 'MSFT', 'MSI', 'NVDA', 'QQQ', 'TXN']
n_days = '10'  # specifies which stock directory we are using so n_days = 5 means looking in data_5
# hashtags to look for in tweets
keyword_list = ['tech', 'technology', 'computer', 'science', 'app', 'laptop', 'device', 'tablet','technews', 'investment', 'stocks', 'bonds', 'nasdaq', 'investor', 'software', 'application', 'device', 'semiconductor']
