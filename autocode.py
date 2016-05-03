#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scipy as sp
from scipy import random as sprd

from lstm import LSTM

data = open('lstm.py', 'r').read()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

seq_len = 25
pos = 0

sample_size = int(data_size / seq_len) - 1
print(data_size)
print(sample_size)
X = sp.zeros((sample_size, seq_len), dtype=int)
T = sp.zeros((sample_size, seq_len), dtype=int)

for i in range(sample_size):
    X[i] = sp.array([char_to_ix[ch] for ch in data[pos: pos + seq_len]])
    T[i] = sp.array([char_to_ix[ch] for ch in data[pos + 1: pos + 1 + seq_len]])
    pos += seq_len

vocab_size = vocab_size
hidden_size = 100
lstm = LSTM(vocab_size, hidden_size)
epoch = 10
sample_len = 800

lstm.train(X, T, 1)
# lstm.train(X, T,
#            epoch=epoch,
#            epoch_end_callback=lambda:
#            print('--------------------------------------------------------------------\n {0}\n --------------------------------------------------------------------'.format(
#                ''.join(ix_to_char[ix] for ix in lstm.sampling(sprd.randint(lstm.vocab_size), sample_len)))))


# def data_loader():
#     i = 0
#     pos = 0
#     while i < sample_size:
#         x = sp.array([char_to_ix[ch] for ch in data[pos: pos + seq_len]])
#         t = sp.array([char_to_ix[ch] for ch in data[pos + 1: pos + 1 + seq_len]])
#         yield (x, t)
#         pos += seq_len
#         i += 1

# lstm.online(data_loader(), epoch=epoch, batch_end_callback=lambda:
#             print('--------------------------------------------------------------------\n {0}\n'.format(
#                 ''.join(ix_to_char[ix] for ix in lstm.sampling(sprd.randint(lstm.vocab_size), sample_len)))))
