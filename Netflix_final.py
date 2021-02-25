from datetime import datetime
import os
import random
import numpy as np
import pandas as pd
from scipy import sparse
from surprise import SVD
from surprise import Reader, Dataset
from sklearn.metrics.pairwise import cosine_similarity


train_data = pd.read_csv("train_data.csv")
test_data = pd.read_csv("test_data.csv")


# if file doesnt exist
# create a sparse matrix from the train data
if not os.path.isfile('sparse_train.npz'):
    sparse_train = sparse.csr_matrix((train_data.Rating.values,
                                      (train_data.UserID.values,
                                       train_data.MovieID.values)),)
    sparse.save_npz("sparse_train.npz", sparse_train)
else:
    sparse_train = sparse.load_npz('sparse_train.npz')


# if file doesnt exist
# create a sparse matrix from the test data
if not os.path.isfile('sparse_test.npz'):
    sparse_test = sparse.csr_matrix((test_data.Rating.values,
                                     (test_data.UserID.values,
                                      test_data.MovieID.values)),)
    sparse.save_npz("sparse_test.npz", sparse_test)
else:
    sparse_test = sparse.load_npz('sparse_test.npz')


print(sparse_train.shape)
print(sparse_test.shape)


# helper function to help compute the similarity matrix
def user_similarity(matrix, few=False, num=100):
    size,size = matrix.shape
    # get the non zero col and rows
    nzrow, nzcol = matrix.nonzero()
    # lists to create a sparse matrix
    num_rows = list()
    num_cols = list()
    data_in = list()
    temp = 0
    for row in nzrow[:num] if few else nzrow:
        temp += 1
        # get similarity for this row with all users
        similar_val = cosine_similarity(matrix.getrow(row), matrix).ravel()
        # the the top num similarirties
        top_sim = similar_val()[-num:]
        sim_vals = similar_val[top_sim]
        # add them to data
        num_rows.extend([row]*num)
        num_cols.extend(top_sim)
        data_in.extend(sim_vals)
    return sparse.csr_matrix((data_in, (num_rows, num_cols)), shape=(size, size))


# if file doesnt exist
# create a similarity matrix for movie and user and save it
if not os.path.isfile('movie_sim.npz'):
    movie_sim = cosine_similarity(X=sparse_train.T, dense_output=False)
    sparse.save_npz("movie_sim.npz", movie_sim)
else:
    movie_sim = sparse.load_npz("movie_sim.npz")


# if file doesnt exist
# create a similarity matrix for movie and user and save it
if not os.path.isfile('user_sim.npz'):
    user_sim = user_similarity(sparse_train, True, 100,)
    sparse.save_npz("user_sim.npz", user_sim)
else:
    user_sim = sparse.load_npz("user_sim.npz")


# helper to extract a sample;
# code taken from and improved upon from stackoverflow, lost the url !
def create_sample(matrix, num_users, num_movies):
    # get the row, col and rate from orignal
    row_idx, col_idx, rate = sparse.find(matrix)
    users = np.unique(row_idx)
    movies = np.unique(col_idx)
    np.random.seed(2345)
    new_users = np.random.choice(users, num_users, replace=False)
    new_movies = np.random.choice(movies, num_movies, replace=False)
    # will be true is in a certain row col in orignal data
    check = np.logical_and(np.isin(row_idx, new_users), np.isin(col_idx, new_movies))
    new_matrix = sparse.csr_matrix((rate[check], (row_idx[check], col_idx[check])),
                                   shape=(max(new_users)+1, max(new_movies)+1))
    return new_matrix


# if file does not exit save it
# create a sample train matrix
if not os.path.isfile("sample_sparse_train.npz"):
    sample_sparse_train = create_sample(sparse_train, 10000, 1000)
    sparse.save_npz("sample_sparse_train.npz", sample_sparse_train)
else:
    sample_sparse_train = sparse.load_npz("sample_sparse_train.npz")


# if file does not exit save it
# create a sample test matrix
if not os.path.isfile("sample_sparse_test.npz"):
    sample_sparse_test = create_sample(sparse_test, 5000, 500)
    sparse.save_npz("sample_sparse_test.npz", sample_sparse_test)
else:
    sample_sparse_test = sparse.load_npz("sample_sparse_test.npz")


# helper method to get averages
def get_average(matrix, of_users):
    # of_user determines whether its for user or movie
    # 1 = user 0 = movie
    if of_users:
        axis = 1
    else:
        axis = 0
    # .A1 is apparently necessary to convert to 1d array
    sum = matrix.sum(axis=axis).A1
    # check if user has rated movie or not
    check = matrix != 0
    # num of ratings user/movie got
    num = check.sum(axis=axis).A1
    # max user and max movie in matrix
    user, movie = matrix.shape
    avg = {i: sum[i] / num[i]
           for i in range(user if of_users else movie)
           if num[i] != 0}
    return avg


# now we will compute all averages
sample_avg = dict()

# we want all the averages and store them in dictionary
# avg movie rating, avg rating, avg user rating
sample_avg['avg'] = sample_sparse_train.sum()/sample_sparse_train.count_nonzero()
sample_avg['user'] = get_average(sample_sparse_train, of_users=True)
sample_avg['movie'] =  get_average(sample_sparse_train, of_users=False)


# create data set using movie, user, rating
# also top 5 movie sim, top 5 user sim, avg, user_avg, movie_avg

# grab already processed files
# creating data took 11 hours
final_train = pd.read_csv('final_train.csv',
                          names=['UserID', 'MovieID', 'Avg',
                                 'User1', 'User2', 'User3', 'User4', 'User5',
                                 'Mov1', 'Mov2', 'Mov3', 'Mov4', 'Mov5',
                                 'UserAvg', 'MovAvg', 'Rating'],
                          header=None)

final_test = pd.read_csv('final_test.csv',
                          names=['UserID', 'MovieID', 'Avg',
                                 'User1', 'User2', 'User3', 'User4', 'User5',
                                 'Mov1', 'Mov2', 'Mov3', 'Mov4', 'Mov5',
                                 'UserAvg', 'MovAvg', 'Rating'],
                          header=None)


# start the modelling

reader = Reader(rating_scale=(1,5))
surprise_traindata = Dataset.load_from_df(final_train[['UserID', 'MovieID', 'Rating']], reader)
trainset = surprise_traindata.build_full_trainset()
testset = list(zip(final_test.UserID.values, final_test.MovieID.values, final_test.Rating.values))

# initiallize the model
svd = SVD(n_factors=100, biased=True, random_state=44, verbose=True)

# train the model using the train set
svd.fit(trainset)

# test the model using test set
svd_pred = svd.test(testset)

actual = np.array([pred.r_ui for pred in svd_pred])
predicted = np.array([pred.est for pred in svd_pred])

rmse = np.sqrt(np.mean((predicted - actual) ** 2))

print('Test Data RMSE: {}'.format(rmse))
