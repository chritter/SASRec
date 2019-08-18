import os
import time
import argparse
import tensorflow as tf
from sampler import WarpSampler
from model import Model
from tqdm import tqdm
from util import *


def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

# Parameters
# ==================================================
# ----------------------- Setup
parser = argparse.ArgumentParser()
parser.add_argument('--dataset',default='Video',type=str)  #required=True)
parser.add_argument('--train_dir',default='training1',type=str)# required=True)

# ----------------------- Hyperparameter
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
# number of self-attention blocks
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=201, type=int)
# number of heads in multi-head attention model
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)

args = parser.parse_args()

# make train directory in data dir
if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
# write training arguments to .txt file for documentation (use mlflow later)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

# creates train, valid, test set, total number of users, items
dataset = data_partition(args.dataset)
[user_train, user_valid, user_test, usernum, itemnum] = dataset
print('number of users in train data: ',len(user_train))
print('number of users in valid data: ',len(user_valid))


# calculates number of batches
num_batch = int(len(user_train) / args.batch_size)
print('Number of batches per epoch: ',num_batch)

# calculates average length of actions
cc = 0.0
for u in user_train:
    cc += len(user_train[u])
print('average sequence length: %.2f' % (cc / len(user_train)))

# for storing training data (surprisingly, no TF functions are used instead)
f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.Session(config=config)

# creates batch iterator for training batches
sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)

# create NN model
model = Model(usernum, itemnum, args)

sess.run(tf.initialize_all_variables())

T = 0.0
t0 = time.time()

try:
    # loop over epochs
    for epoch in range(1, args.num_epochs + 1):
        # loop over batches
        for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            # get batches
            u, seq, pos, neg = sampler.next_batch()
            # do the training
            auc, loss, _ = sess.run([model.auc, model.loss, model.train_op],
                                    {model.u: u, model.input_seq: seq, model.pos: pos, model.neg: neg,
                                     model.is_training: True})
        # evaluate every 20 epochs
        if epoch % 20 == 0:
            t1 = time.time() - t0
            T += t1
            print('Evaluating')

            # calculate Hit@10 rate and NDCG@10 for test set
            t_test = evaluate(model, dataset, args, sess)

            # calculate Hit@10 rate and NDCG@10 for validation set
            t_valid = evaluate_valid(model, dataset, args, sess)

            print()
            print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)' % (
            epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))

            # write out metrics
            f.write(str(t_valid) + ' ' + str(t_test) + '\n')
            f.flush()
            t0 = time.time()
except:
    sampler.close()
    f.close()
    exit(1)

f.close()
sampler.close()
print("Done")
