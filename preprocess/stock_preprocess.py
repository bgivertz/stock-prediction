import os
import csv
from preprocess.stocks import *

def create_stock(name, data):
    prices = []
    for d in data:
        #first 7 elements of every row in stock csv should correspond with various prices
        p = Price(d[0], d[1], d[2], d[3], d[4], d[6])
        prices.append(p)
    return Stock(name, prices)

#creates a list of every valid row in the csv file
def parse_csv(path):
    data = []
    skipped_rows = [] #holds rows that wont be added to parsed stock csv
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
    return [ os.path.join(path, filename) for filename in filenames if filename.endswith( '.csv' ) ]

def generate_stock_csvs(path, verbose):
    # gets names of all stock csv files in data directory
    stock_csv_files = get_stock_csv_files(path)
    prev_days = [5, 10, 15, 20, 25, 30]
    for n in prev_days:
        new_dir = os.path.join(path, f'data_{n}')
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)

        for csv_file in stock_csv_files:
            (file_name, extension) = os.path.splitext(os.path.basename(csv_file))
            # creates a list of every (non null) row in stock csv files
            data, skipped_rows = parse_csv(csv_file)

            #only want to print skipped rows once
            if verbose == True:
                print('\nfor day ' + str(n) + ' the skipped rows were:')
                print('\t' + '\n\t'.join(skipped_rows))



            # creates a stock object that stores a list of all the prices (a list of Price objects)
            # on each date
            stock = create_stock(file_name, data)
            # creates a new csv file, but this will hold only the cleaned stock information a multiple
            # of n days in the past
            new_file_name = f'{file_name}-{n}{extension}'
            new_path = os.path.join(new_dir, new_file_name)
            stock.to_csv(new_path, n)

    stock = create_stock("", data)


