import numpy as np
from multiprocessing import Process, Queue


def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    '''
    Create batch of user samples
    :param user_train:
    :param usernum:
    :param itemnum:
    :param batch_size:
    :param maxlen:
    :param result_queue:
    :param SEED:
    :return:
    '''
    def sample():
        '''
        Create one user sample, prepare sequence, incl. negative sample and positional embedding. Add to queue for output
        :return:
        '''
        # select random user id
        user = np.random.randint(1, usernum + 1)
        # while number of actions are 1 or less, sample other user
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

        # create sequence with full zero padding, later fill values
        seq = np.zeros([maxlen], dtype=np.int32) # item embedding
        pos = np.zeros([maxlen], dtype=np.int32) # positional embedding P
        neg = np.zeros([maxlen], dtype=np.int32) # random negative sample sequence as long as item sequence
        nxt = user_train[user][-1]
        idx = maxlen - 1

        # this amounts to adding items to the right, and the padding will be on the left (paper)
        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        return (user, seq, pos, neg)

    np.random.seed(SEED)

    # create batch by sampling
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    #
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        '''
        Start n_workers processes, each executing sample_function
        :param User:
        :param usernum:
        :param itemnum:
        :param batch_size:
        :param maxlen:
        :param n_workers:
        '''
        # Create a queue object with a given maximum size.
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()
