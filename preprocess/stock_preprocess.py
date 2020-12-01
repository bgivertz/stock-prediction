import os
import csv
from preprocess.stocks import *
import yfinance as yf
from preprocess import config
import shutil
from datetime import date, timedelta


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
        #skip header
        next(reader)
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


def convert_to_date_object(date_string):
    date_format = date_string.split('-')
    # convert day to date format, and get day after current day
    return date(int(date_format[0]), int(date_format[1]), int(date_format[2]))


def pull_csvs_from_yahoo_finance(path):
    for stock_listing_symbol in config.stock_listing_symbols:
        data_as_dataframe = yf.download(stock_listing_symbol, start=config.date_range_start, end=config.date_range_end,
                                        progress=False)
        data_as_dataframe.to_csv(path + '/' + stock_listing_symbol + '.csv')


def create_uniform_skipped_rows(path, skipped_rows):
    prev_days = [5, 10, 15, 20, 25, 30]
    for n in prev_days:
        # get csvs from each stock data subdirectory
        stock_sub_dir = path + '/data_' + str(n)
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
                        else:
                            print(row_num)
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
    prev_days = [5, 10, 15, 20, 25, 30]
    for n in prev_days:

        new_dir = os.path.join(path, f'data_{n}')
        # remove data in data directories already existing, and generate them anew
        if os.path.exists(new_dir):
            shutil.rmtree(new_dir)
        os.makedirs(new_dir)

        # for each stock
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
            stock.to_csv(new_path, n)

    #all stocks have the same dates in csv
    create_uniform_skipped_rows(path, skipped_rows)