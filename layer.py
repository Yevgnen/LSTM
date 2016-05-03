#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scipy as sp

from activation import Sigmoid, Softmax, Tanh

ACTIVATION_MAP = {'tanh': Tanh(), 'sigmoid': Sigmoid(), 'softmax': Softmax()}


class LSTMLayer(object):
    def __init__(self, activation='tanh'):
        self.activation = activation
        self.a = None
        self.h = None
        self.y = None

    def activate(self, x):
        return ACTIVATION_MAP[self.activation].eval(x)

    def backward(self):
        return ACTIVATION_MAP[self.activation].gradient(self.a)

    def loss(self, t):
        return ACTIVATION_MAP[self.activation].loss(t, self.y)


class HiddenLayer(LSTMLayer):
    def __init__(self, hidden_size=10, gate_activation='sigmoid', input_activation='tanh', output_activation='tanh'):
        self.hidden_size = hidden_size
        self.gate_activation = gate_activation
        self.input_activation = input_activation
        self.output_activation = output_activation
        self.gate_activate = ACTIVATION_MAP[self.gate_activation]
        self.input_activate = ACTIVATION_MAP[self.input_activation]
        self.output_activate = ACTIVATION_MAP[self.output_activation]
        self.c = sp.zeros(self.hidden_size)
        self.h = sp.zeros(self.hidden_size)

    def forward(self, Wz, Wi, Wf, Wo, Rz, Ri, Rf, Ro, pi, pf, po, bz, bi, bf, bo, x, prev_c, prev_h):
        # Block input
        self.zbar = Wz[:, x] + sp.dot(Rz, prev_h) + bz
        self.z = self.input_activate.eval(self.zbar)

        # Input gate
        self.ibar = Wi[:, x] + sp.dot(Ri, prev_h) + pi * prev_c + bi
        self.i = self.gate_activate.eval(self.ibar)

        # Forget gate
        self.fbar = Wf[:, x] + sp.dot(Rf, prev_h) + pf * prev_c + bf
        self.f = self.gate_activate.eval(self.fbar)

        # Cell
        self.c = self.z * self.i + prev_c * self.f
        self.a = self.output_activate.eval(self.c)

        # Output gate
        self.obar = Wo[:, x] + sp.dot(Ro, prev_h) + po * self.c + bo
        self.o = self.gate_activate.eval(self.obar)

        # Block output
        # The origin paper `LSTM: A Search Space Odyssey` use `y` to denote block output, but here use `h` instead.
        self.h = self.a * self.o

    def backward(self):
        return (self.input_activate.gradient(self.zbar),
                self.gate_activate.gradient(self.ibar),
                self.gate_activate.gradient(self.fbar),
                self.gate_activate.gradient(self.obar),
                self.output_activate.gradient(self.c))


class OutputLayer(LSTMLayer):
    def __init__(self, hidden_size=10, activation='softmax'):
        super(OutputLayer, self).__init__(activation)
        self.hidden_size = hidden_size

    def forward(self, V, h, c):
        self.a = sp.dot(V, h) + c
        self.y = self.activate(self.a)
