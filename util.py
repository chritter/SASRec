import sys
import copy
import random
import numpy as np
from collections import defaultdict


def data_partition(fname):
    '''
    Reads file in data directory with name fname.
     Expect format: userID itemID, both ID's start from 1
    Separates data into train, valid and test set
    :param fname:
    :return:
    '''
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        # number of actions per user
        nfeedback = len(User[user])
        # if actions less than 3, only add to training set
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            # add second last item to validation set
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            # add last item to test set
            user_test[user].append(User[user][-1])

    return [user_train, user_valid, user_test, usernum, itemnum]


def evaluate(model, dataset, args, sess):
    '''
    Evaluate Hit@10 and NDGC@10 on test data
    :param model:
    :param dataset: whole dataset
    :param args:
    :param sess:
    :return:
    '''

    # make copy of whole data set
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    # why this limit?
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)


    for u in users:
        # if user not in training set or test set, skip
        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1

        # why is this being done?
        seq[idx] = valid[u][0]

        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)

        # get item from test set
        item_idx = [test[u][0]]

        # randomly sample 100 negative items (paper)
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            # select new item if item already present in user series
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        # get predictions for 101 items
        predictions = -model.predict(sess, [u], [seq], item_idx)
        predictions = predictions[0]

        # get rank of items
        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        # calculate metrics for Metrics@k with k=10
        if rank < 10:
            # NDCG metric
            NDCG += 1 / np.log2(rank + 2)
            # Hit Rate
            HT += 1
        if valid_user % 100 == 0:
            print('.',
            sys.stdout.flush())

    return NDCG / valid_user, HT / valid_user


def evaluate_valid(model, dataset, args, sess):
    '''
    Evaluate Hit@10 and NDGC@10 on evaluation data
    :param model:
    :param dataset:
    :param args:
    :param sess:
    :return:
    '''

    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0

    # why?
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)

    for u in users:

        # if user not in training set or valid set, skip
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1

        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)

        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        # get predictions
        predictions = -model.predict(sess, [u], [seq], item_idx)
        predictions = predictions[0]

        # calculate ranks for items
        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        # metrics
        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.',
            sys.stdout.flush())

    return NDCG / valid_user, HT / valid_user
