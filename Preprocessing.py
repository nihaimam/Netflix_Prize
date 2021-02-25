from datetime import datetime
import os
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csr_matrix

# to keep track of how long the entire program take
start_time = datetime.now()

# create a big data file by merging all the data given
# if file doesn't exist
if not os.path.isfile('data.csv'):
    # open a new file
    new_file = open('data.csv','a')
    row = list()
    data_files = ['netflix-prize-data/combined_data_1.txt',
                  'netflix-prize-data/combined_data_2.txt',
                  'netflix-prize-data/combined_data_3.txt',
                  'netflix-prize-data/combined_data_4.txt']
    # read in from every file in to one big file
    for file in data_files:
        # for every line in the file
        with open(file) as f:
            for line in f:
                # get rid of any white spaces
                line = line.strip()
                # movie id's end with : eg 1: or 43:
                if line.endswith(':'):
                    # save the number of movie id for later use
                    movie_id = line.replace(':', '')
                else:
                    # merge the rating and the movie id together
                    row = [x for x in line.split(',')]
                    row.insert(0, movie_id)
                    # insert the data into the csw
                    new_file.write(','.join(row))
                    new_file.write('\n')
    new_file.close()

# 3 min

# create a data frame from the combined data
dataframe = pd.read_csv('data.csv', sep=',', names=['MovieID', 'UserID', 'Rating', 'Date'])
dataframe.Date = pd.to_datetime(dataframe.Date)

# sort the dataframe according to time
# movie id and user id are not good sorting choices
dataframe.sort_values(by='Date')

# 5 min

# create a train data frame and save it
if not os.path.isfile('train_data.csv'):
    # using the first 80% of the dataframe
    train_data = dataframe.iloc[:int(dataframe.shape[0] * 0.80)]
    train_data.to_csv("train_data.csv", index=False)
else:
    train_data = pd.read_csv("train_data.csv")

# create a test data frame and save it
if not os.path.isfile('test_data.csv'):
    # using the last 20% of the dataframe
    test_data = dataframe.iloc[:int(dataframe.shape[0] * 0.80):]
    test_data.to_csv("test_data.csv", index=False)
else:
    test_data = pd.read_csv("test_data.csv")

print(datetime.now()-start_time)
