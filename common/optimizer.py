# !user/bin/python
# -*- coding: UTF-8 -*-

import numpy as np

class SGD:
    def __init__(self, lr = 0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]

class Momentum:
    def __init__(self, lr = 0.1, momentum = 0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v == None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] + (1- self.momentum) * grads[key]
            params[key] -= self.lr * self.v[key]

class AdaGrad:
    def __init__(self, lr = 0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h == None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] ** 2
            params[key] -= (self.lr * grads[key] / np.sqrt(self.h[key] + 1e-7))

class RMSprop:
    def __init__(self, lr = 0.01, decay = 0.9):
        self.lr = lr
        self.h = None
        self.decay = decay

    def update(self, params, grads):
        if self.h == None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] = self.decay * self.h[key] + (1 - self.decay) * (grads[key] ** 2)
            params[key] -= (self.lr * grads[key] / np.sqrt(self.h[key] + 1e-7))

class Adam:
    def __init__(self, lr = 0.001, beta1 = 0.9, beta2 = 0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = None
        self.v = None
        self.t = 0

    def update(self, params, grads):
        if self.m is None and self. v is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.t += 1
        lr_t = self.lr * np.sqrt(1 - self.beta2 ** self.t) /(1 - self.beta1 ** self.t)
        for key in params.keys():
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)
            params[key] -= (lr_t * self.m[key] / np.sqrt(self.v[key] + 1e-7))
