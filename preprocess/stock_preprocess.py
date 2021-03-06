import os
import csv
from preprocess.stocks import *
import yfinance as yf
import preprocess.config as config
import shutil
import numpy as np
import datetime as dt
from sklearn.preprocessing import MinMaxScaler


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
    skipped_rows = set()
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row_num, row in enumerate(reader):
            try:
                row_data = [row[0]]
                row_data.extend(list(map(lambda n: float(n), row[1:])))
                data.append(row_data)
            except:
                skipped_rows.add(row_num)
    return data, skipped_rows


def get_stock_csv_files(path):
    filenames = os.listdir(path)
    return [os.path.join(path, filename) for filename in filenames if filename.endswith('.csv')]


def pull_csvs_from_yahoo_finance(path):
    for stock_listing_symbol in config.stock_listing_symbols:
        start_date = dt.datetime.strptime(config.date_range_start, '%Y-%m-%d')
        delt = dt.timedelta(100)
        start = (start_date - delt).strftime('%Y-%m-%d')

        data_as_dataframe = yf.download(stock_listing_symbol, start=start, end=config.date_range_end,
                                        progress=False)
        data_as_dataframe.to_csv(path + '/' + stock_listing_symbol + '.csv')


'''
makes every stock within a data_# subdirectory
have the same number of rows (and dates)
'''


def create_uniform_skipped_rows(path, skipped_rows):

    # get csvs from each stock data subdirectory
    stock_sub_dir = path + '/data_' + str(config.n_days)
    for csv_file in get_stock_csv_files(stock_sub_dir):
        # open each csv file in subdir
        with open(csv_file, 'r') as current_file:
            # read each row, and if it is not in skipped rows add it to a cleaned
            # version of the csv
            csv_reader = csv.reader(current_file)
            with open(os.path.splitext(csv_file)[0] + '-clean.csv', 'w') as csvfile:
                csvwriter = csv.writer(csvfile)
                for row_num, row in enumerate(csv_reader):
                    if row_num not in skipped_rows:
                        csvwriter.writerow(row)
        os.remove(csv_file)
        os.rename(os.path.splitext(csv_file)[0] + '-clean.csv', csv_file)


def generate_stock_csvs(path, verbose):
    skipped_rows = set()  # holds rows that wont be added to parsed stock csv

    # delete any existing csv files, so as to only retrieve needed csvs
    existing_csv_files = get_stock_csv_files(path)
    for file in existing_csv_files:
        os.remove(file)

    # gets needed csvs
    pull_csvs_from_yahoo_finance(path)

    # gets names of all stock csv files in data directory
    stock_csv_files = get_stock_csv_files(path)
    n = int(config.n_days)

    new_dir = os.path.join(path, f'data_{n}')
    # remove data in data directories already existing, and generate them anew
    if os.path.exists(new_dir):
        shutil.rmtree(new_dir)
    os.makedirs(new_dir)

    for csv_file in stock_csv_files:
        (file_name, extension) = os.path.splitext(os.path.basename(csv_file))
        # creates a list of every (non null) row in stock csv files
        data, one_stock_skipped_rows = parse_csv(csv_file)
        skipped_rows = one_stock_skipped_rows | skipped_rows
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
        stock.to_csv(new_path, n, config.date_range_start)

    # all stocks have the same dates in csv
    create_uniform_skipped_rows(path, skipped_rows)

    stock_sub_dir = path + '/data_' + str(config.n_days)
    return get_stock_csv_files(stock_sub_dir)


def csv_to_vector(path):
    stock_vector = np.genfromtxt(path, delimiter=',')
    return stock_vector[:, 2:] #skips first two columns (date and stock name)
