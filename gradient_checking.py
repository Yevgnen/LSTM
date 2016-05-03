#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scipy import random as sprd

from lstm import LSTM

n = 10
x = sprd.permutation(n)
t = list(reversed(x))
hidden_size = 4
vocab_size = n

lstm = LSTM(vocab_size, hidden_size)
lstm.check_gradient(x, t)
