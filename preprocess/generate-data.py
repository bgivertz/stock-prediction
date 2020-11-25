import sys  
import os
import csv
from stocks import *

def create_stock(name, data):
    prices = []
    for d in data:
        p = Price(d[0], d[1], d[2], d[3], d[4], d[6])
        prices.append(p)
    return Stock(name, prices)

def parse_csv(path):
    data = []
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            try:
                row_data = [row[0]]
                row_data.extend(list(map(lambda n: float(n), row[1:])))
                data.append(row_data)
            except:
                print(f"Did not add row: {row}")
    return data

def get_csv_files(path):
    filenames = os.listdir(path)
    return [ os.path.join(path, filename) for filename in filenames if filename.endswith( '.csv' ) ]


def parse_args():
    if len(sys.argv) < 2:
        print("USAGE: python generate-data.py <data dir>")
        exit()
    
    path = sys.argv[1]
    if os.path.isdir(path):
        return path
    else:
        print("ERROR: path must be a directory")
        exit()

def main():
    path = parse_args()
    csv_files = get_csv_files(path)
    prev_days = [5, 10, 15, 20, 25, 30]

    for n in prev_days:
        new_dir = os.path.join(path, f'data_{n}')
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)

        for csv_file in csv_files:
            (file_name, extension) = os.path.splitext(os.path.basename(csv_file))
            data = parse_csv(csv_file)
            stock = create_stock(file_name, data)

            new_file_name = f'{file_name}-{n}{extension}'
            new_path = os.path.join(new_dir, new_file_name)
            stock.to_csv(new_path, n)

    stock = create_stock("", data)

    

if __name__ == '__main__':
    main()