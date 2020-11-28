import os
import csv
from preprocess.stocks import *
import yfinance as yf
from preprocess import config
import shutil
import numpy as np


def create_stock(name, data):
    prices = []
    for d in data:
        # first 7 elements of every row in stock csv should correspond with various prices
        p = Price(d[0], d[1], d[2], d[3], d[4], d[6])
        prices.append(p)
    return Stock(name, prices)


# creates a list of every valid row in the csv file
def parse_csv(path):
    data = []
    skipped_rows = []  # holds rows that wont be added to parsed stock csv
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row_num, row in enumerate(reader):
            try:
                row_data = [row[0]]
                row_data.extend(list(map(lambda n: float(n), row[1:])))
                data.append(row_data)
            except:
                skipped_rows.append(str(row_num) + ': ' + str(row))
    return data, skipped_rows


def get_stock_csv_files(path):
    filenames = os.listdir(path)
    return [os.path.join(path, filename) for filename in filenames if filename.endswith('.csv')]


def pull_csvs_from_yahoo_finance(path):
    for stock_listing_symbol in config.stock_listing_symbols:
        data_as_dataframe = yf.download(stock_listing_symbol, start=config.date_range_start, end=config.date_range_end,
                                        progress=False)
        data_as_dataframe.to_csv(path + '/' + stock_listing_symbol + '.csv')


def generate_stock_csvs(path, verbose):

    #delete any existing csv files, so as to only retrieve needed csvs
    existing_csv_files = get_stock_csv_files(path)
    for file in existing_csv_files:
        os.remove(file)

    #gets needed csvs
    pull_csvs_from_yahoo_finance(path)

    # gets names of all stock csv files in data directory
    stock_csv_files = get_stock_csv_files(path)
    prev_days = [5, 10, 15, 20, 25, 30]
    for n in prev_days:

        new_dir = os.path.join(path, f'data_{n}')
        # remove data in data directories already existing, and generate them anew
        shutil.rmtree(new_dir)
        os.makedirs(new_dir)

        for csv_file in stock_csv_files:
            (file_name, extension) = os.path.splitext(os.path.basename(csv_file))
            # creates a list of every (non null) row in stock csv files
            data, skipped_rows = parse_csv(csv_file)

            # only want to print skipped rows if verbose tag was used
            if verbose == True:
                print('\nfor day ' + str(n) + ' the skipped rows were:')
                print('\t' + '\n\t'.join(skipped_rows))

            # creates a stock object that stores a list of all the prices (a list of Price objects)
            # on each date
            stock = create_stock(file_name, data)
            # creates a new csv file, but this will hold only the cleaned stock and information about
            # that stock using the past n days
            new_file_name = f'{file_name}-{n}{extension}'
            new_path = os.path.join(new_dir, new_file_name)
            stock.to_csv(new_path, n)

    stock = create_stock("", data)

'''
    looks in stock data directory, and converts all stock data to 3d vector of shape
    (num_stocks, num days, stock parameter size)
'''
def vectorize_stock_csv(path):
    stock_directory = os.path.join(path, f'data_{config.n_days}')
    filenames = os.listdir(stock_directory)
    all_stock_data = []
    for file_num, file in enumerate(filenames):
        stocks_by_day = [] #shape = (num days,  15 (number of properties of stock))
        with open(stock_directory + '/' + file, 'r') as current_file:
            reader = csv.reader(current_file)
            for row in reader:
                stocks_by_day.append(row)
        numpy_stocks_by_day = np.array(stocks_by_day)[1:, 2:]
        all_stock_data.append(numpy_stocks_by_day)
    all_stock_data = np.array(all_stock_data)
    return all_stock_data