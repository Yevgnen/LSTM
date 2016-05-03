#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scipy as sp
from scipy.special import expit


class Sigmoid(object):
    def eval(self, x):
        return expit(x)

    def gradient(self, x):
        y = self.eval(x)
        # FIXME
        return y * (1 - y)


class Tanh(object):
    def eval(self, x):
        return sp.tanh(x)

    def gradient(self, x):
        y = self.eval(x)
        return 1 - y**2


class Softmax(object):
    def eval(self, x):
        x = sp.exp(x - x.max(axis=0))
        return x / x.sum(axis=0)

    def gradient(self, x):
        y = self.eval(x)
        return sp.diag(y) - sp.outer(y, y)

    def loss(self, t, y, eps=1e-15):
        y = sp.clip(y, eps, 1 - eps)
        return -sp.sum(t * sp.log(y))
