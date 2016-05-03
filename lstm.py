#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path
import pickle
from datetime import datetime

import scipy as sp
from scipy import random as sprd

from layer import HiddenLayer, OutputLayer


class LSTM(object):
    def __init__(self, vocab_size, hidden_size, bptt_truncate=4, clip=5, save_path=None):
        self.vocab_size = vocab_size    # The size of the dictionary.
        self.hidden_size = hidden_size

        # Initialize the hidden layer
        # Initialize the input weights
        self.Wz = sprd.uniform(-sp.sqrt(1. / vocab_size), sp.sqrt(1. / vocab_size), (hidden_size, vocab_size))
        self.Wi = sprd.uniform(-sp.sqrt(1. / vocab_size), sp.sqrt(1. / vocab_size), (hidden_size, vocab_size))
        self.Wf = sprd.uniform(-sp.sqrt(1. / vocab_size), sp.sqrt(1. / vocab_size), (hidden_size, vocab_size))
        self.Wo = sprd.uniform(-sp.sqrt(1. / vocab_size), sp.sqrt(1. / vocab_size), (hidden_size, vocab_size))
        # Initialize the recurrent weights
        self.Rz = sprd.uniform(-sp.sqrt(1. / hidden_size), sp.sqrt(1. / hidden_size), (hidden_size, hidden_size))
        self.Ri = sprd.uniform(-sp.sqrt(1. / hidden_size), sp.sqrt(1. / hidden_size), (hidden_size, hidden_size))
        self.Rf = sprd.uniform(-sp.sqrt(1. / hidden_size), sp.sqrt(1. / hidden_size), (hidden_size, hidden_size))
        self.Ro = sprd.uniform(-sp.sqrt(1. / hidden_size), sp.sqrt(1. / hidden_size), (hidden_size, hidden_size))
        # Initialize the peephole weights
        self.pi = sp.zeros(hidden_size)
        self.pf = sp.zeros(hidden_size)
        self.po = sp.zeros(hidden_size)
        # Initialize the bias weights
        self.bz = sp.zeros(hidden_size)
        self.bi = sp.zeros(hidden_size)
        self.bf = sp.zeros(hidden_size)
        self.bo = sp.zeros(hidden_size)

        # Initialize the output layer
        self.V = sprd.uniform(-sp.sqrt(1. / hidden_size), sp.sqrt(1. / hidden_size), (vocab_size, hidden_size))
        self.c = sp.zeros(self.vocab_size)

        self.param_names = ['Wz', 'Wi', 'Wf', 'Wo', 'Rz', 'Ri', 'Rf', 'Ro',
                            'pi', 'pf', 'po', 'bz', 'bi', 'bf', 'bo', 'V', 'c']

        self.bptt_truncate = bptt_truncate
        self.clip = clip
        self.save_path = save_path

    def save(self, file='pickle.dat'):
        data = [getattr(self, param) for param in self.param_names]
        with open(file, 'wb') as f:
            pickle.dump(data, f)

    def load(self, file='pickle.dat'):
        with open(file, 'rb') as f:
            [setattr(self, param, value) for (param, value) in zip(self.param_names, pickle.load(f))]

    def compute_loss(self, x, t):
        """Compute loss of a sample."""
        return self.forward_propagation(x, t)[1]

    def compute_total_loss(self, X, T):
        """Compute the total loss of all samples."""
        loss = 0.0
        sample_size = len(X)
        for i in range(sample_size):
            loss += self.compute_loss(X[i], T[i])

        return loss / sample_size

    def sampling(self, x, len):
        cell = self.create_cell()
        sample = []
        for t in range(len):
            cell = self.forward_time(x, cell)
            x = sprd.choice(range(self.vocab_size), p=cell[-1].y)
            sample.append(x)

        return sample

    def create_cell(self):
        # FIXME: Should create cells base on network structure
        hidden = HiddenLayer(self.hidden_size)
        output = OutputLayer(self.hidden_size)

        return (hidden, output)

    def forward_time(self, ix, prev_cell):
        """Forward in time t given current input and previous hidden cell state."""
        # FIXME: There maybe multiple layers here.
        # Compute the hidden state
        (hidden, output) = self.create_cell()

        hidden.forward(self.Wz, self.Wi, self.Wf, self.Wo, self.Rz, self.Ri, self.Rf, self.Ro, self.pi, self.pf,
                       self.po, self.bz, self.bi, self.bf, self.bo, ix, prev_cell[0].c, prev_cell[0].h)
        # Compute the output
        output = OutputLayer(self.hidden_size)
        output.forward(self.V, hidden.h, self.c)

        return (hidden, output)

    def forward_propagation(self, x, t):
        """Forward Propagation of a single sample."""
        tau = len(x)
        loss = 0.0
        cells = [None for i in range(tau)]

        cell = self.create_cell()
        for i in range(tau):
            cell = self.forward_time(x[i], cell)
            one_hot_t = sp.zeros(self.vocab_size)
            one_hot_t[t[i]] = 1
            loss += cell[-1].loss(one_hot_t)
            cells[i] = cell

        return (cells, loss)

    def bptt(self, x, t, cells):
        """Back propagation throuth time of a sample.

        Reference: [1] LSTM: A Search Space Odyssey, Klaus Greff, Rupesh Kumar Srivastava, Jan Koutník,
                       Bas R. Steunebrink, Jürgen Schmidhuber, http://arxiv.org/abs/1503.04069
        """
        dWz = sp.zeros_like(self.Wz)
        dWi = sp.zeros_like(self.Wi)
        dWf = sp.zeros_like(self.Wf)
        dWo = sp.zeros_like(self.Wo)

        dRz = sp.zeros_like(self.Rz)
        dRi = sp.zeros_like(self.Ri)
        dRf = sp.zeros_like(self.Rf)
        dRo = sp.zeros_like(self.Ro)

        dpi = sp.zeros_like(self.pi)
        dpf = sp.zeros_like(self.pf)
        dpo = sp.zeros_like(self.po)

        dbz = sp.zeros_like(self.bz)
        dbi = sp.zeros_like(self.bi)
        dbf = sp.zeros_like(self.bf)
        dbo = sp.zeros_like(self.bo)

        dV = sp.zeros_like(self.V)
        dc = sp.zeros_like(self.c)

        tau = len(x)

        dcbar = sp.zeros(self.hidden_size)
        next_dzbar = sp.zeros(self.hidden_size)
        next_dibar = sp.zeros(self.hidden_size)
        next_dfbar = sp.zeros(self.hidden_size)
        next_dobar = sp.zeros(self.hidden_size)

        for i in range(tau - 1, -1, -1):
            # FIXME:
            # 1. Should not use cell[i] since there maybe multiple hidden layers.
            # 2. Using exponential family as output should not be specified.
            ix = x[i]
            one_hot_t = sp.zeros(self.vocab_size)
            one_hot_t[t[i]] = 1

            # Cell of time i
            cell = cells[i]
            # Hidden layer of current cell
            hidden = cell[0]
            # Output layer of current cell
            output = cell[-1]
            # Hidden layer of time i + 1
            prev_hidden = cells[i - 1][0] if i - 1 >= 0 else None
            # Hidden layer of time i - 1
            next_hidden = cells[i + 1][0] if i + 1 < tau else None

            # Error of current time i
            (gzbar, gibar, gfbar, gobar, gc) = hidden.backward()
            # Error or information of time i + 1 or i - 1
            prev_c = prev_hidden.c if prev_hidden is not None else sp.zeros(self.hidden_size)
            next_f = next_hidden.f if next_hidden is not None else sp.zeros(self.hidden_size)

            # FIXME: The error function should not be specified here
            # The actural and slow epxression for `da` is: da = sp.dot(output.backward().T, -one_hot_t / output.y)
            # The order of evaluating the deltas is IMPORTANT!
            output_da = output.y - one_hot_t
            dh = sp.dot(self.V.T, output_da) + sp.dot(self.Rz.T, next_dzbar) + sp.dot(self.Ri.T, next_dibar) + sp.dot(
                self.Rf.T, next_dfbar) + sp.dot(self.Ro.T, next_dobar)
            dobar = dh * hidden.a * gobar
            dcbar = dh * hidden.o * gc + self.po * dobar + self.pi * next_dibar + self.pf * next_dfbar + dcbar * next_f
            dfbar = dcbar * prev_c * gfbar
            dibar = dcbar * hidden.z * gibar
            dzbar = dcbar * hidden.i * gzbar

            # Gradient back propagation through time
            dbz += dzbar
            dbi += dibar
            dbf += dfbar
            dbo += dobar

            dpi += hidden.c * next_dibar
            dpf += hidden.c * next_dfbar
            dpo += hidden.c * dobar

            dWz[:, ix] += dzbar
            dWi[:, ix] += dibar
            dWf[:, ix] += dfbar
            dWo[:, ix] += dobar

            dRz += sp.outer(next_dzbar, hidden.h)
            dRi += sp.outer(next_dibar, hidden.h)
            dRf += sp.outer(next_dfbar, hidden.h)
            dRo += sp.outer(next_dobar, hidden.h)

            dV += sp.outer(output_da, hidden.h)
            dc += output_da

            # Save current information for time i - 1
            next_dzbar = dzbar
            next_dibar = dibar
            next_dfbar = dfbar
            next_dobar = dobar

            grads = (dWz, dWi, dWf, dWo, dRz, dRi, dRf, dRo, dpi, dpf, dpo, dbz, dbi, dbf, dbo, dV, dc)
            for grad in grads:
                sp.clip(grad, -self.clip, self.clip, out=grad)

        return grads

    def sgd_step(self, x, t, learning_rate):
        """Process SGD using one single sample."""
        # Forward propagation
        (cells, loss) = self.forward_propagation(x, t)

        # Backward progation throuth time
        grads = self.bptt(x, t, cells)

        # Gradient update
        for (dparam, param) in zip(
                grads,
                self.param_names):
            setattr(self, param, getattr(self, param) - learning_rate * dparam)

    def train(self, X, T, epoch=100, learning_rate=0.1, lr_factor=0.9, epoch_end_callback=None):
        """Train the network by SGD."""
        losses = sp.zeros(epoch)
        for j in range(epoch):
            # Scan the full training set
            for i, (x, t) in enumerate(zip(X, T)):
                self.sgd_step(x, t, learning_rate)

            timestr = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            losses[j] = self.compute_total_loss(X, T)
            print('{0}: After epoch={1}, loss={2}, lr={3}.'.format(timestr, j + 1, losses[j], learning_rate))

            # Adjust the learning rate if the loss increased
            if j > 0 and losses[j] > losses[j - 1]:
                learning_rate *= lr_factor

            # Save params of each epoch
            if (self.save_path is not None):
                self.save(os.path.join(self.save_path, 'epoch{0}.dat'.format(j + 1)))

            if epoch_end_callback is not None and callable(epoch_end_callback):
                epoch_end_callback()

    def numerical_gradient(self, x, t, eps=1e-5):
        grad = {}

        for name in self.param_names:
            A = getattr(self, name)
            dA = sp.zeros_like(A)
            if (A.ndim == 2):
                (row, col) = A.shape
                for i in range(row):
                    for j in range(col):
                        aij = A[i, j]

                        A[i, j] = aij + eps
                        setattr(self, name, A)
                        lh = self.compute_loss(x, t)

                        A[i, j] = aij - eps
                        setattr(self, name, A)
                        lo = self.compute_loss(x, t)

                        dA[i, j] = (lh - lo) / (2.0 * eps)
                        A[i, j] = aij
            elif (A.ndim == 1):
                length = A.shape[0]
                for i in range(length):
                    ai = A[i]

                    A[i] = ai + eps
                    setattr(self, name, A)
                    lh = self.compute_loss(x, t)

                    A[i] = ai - eps
                    setattr(self, name, A)
                    lo = self.compute_loss(x, t)

                    dA[i] = (lh - lo) / (2.0 * eps)
                    A[i] = ai
            grad['d{0}'.format(name)] = dA

        return grad

    def check_gradient(self, x, t):
        (cells, _) = self.forward_propagation(x, t)
        (dWz, dWi, dWf, dWo, dRz, dRi, dRf, dRo, dpi, dpf, dpo, dbz, dbi, dbf, dbo, dV, dc) = self.bptt(x, t, cells)
        grad = self.numerical_gradient(x, t)

        for (dparam, param) in zip(
            (dWz, dWi, dWf, dWo, dRz, dRi, dRf, dRo, dpi, dpf, dpo, dbz, dbi, dbf, dbo, dV, dc), self.param_names):
            print('Gradient checking: max|d{0} - nd{0}| = {1}'.format(
                param, sp.absolute(dparam - grad['d' + param]).max()))
