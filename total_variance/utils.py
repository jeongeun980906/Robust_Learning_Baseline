from datetime import datetime
from functools import lru_cache
from typing import Iterable,Tuple
from scipy.stats import kendalltau
import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms

from data.cifar import CIFAR10,CIFAR100
from data.mnist import MNIST
from total_variance.core import (
    NoTransition, CategoricalTransition, DirichletTransition,
    no_regularization, tv_regularization,
    get_train_step,)

def kendall_tau(wm,tm):
    labels=wm.shape[0]
    res=0
    for i in range(labels):
        tau, p_value=kendalltau(wm[i,:],tm[i,:])
        res+=tau
    return res/labels

def diag_matrix(n: int, diagonal: float, off_diagonal: float) -> torch.Tensor:
    return off_diagonal * torch.ones(n, n) + (diagonal - off_diagonal) * torch.eye(n, n)


def categorical_transition(device, num_classes, num_iter_warmup, num_iter_total, diagonal, off_diagonal, lr):
    init_matrix = diag_matrix(num_classes, diagonal=diagonal, off_diagonal=off_diagonal).to(device)
    optim_matrix = lambda params: optim.Adam(params, lr=lr)
    lr_lambda = lambda i: np.interp([i], [0, num_iter_warmup, num_iter_total], [0, 1, 0])[0]
    sched_matrix = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return CategoricalTransition(init_matrix, optim_matrix, sched_matrix)

def dirichlet_transition(device, num_classes, diagonal, off_diagonal, betas):
    init_matrix = diag_matrix(num_classes, diagonal=diagonal, off_diagonal=off_diagonal).to(device)
    return DirichletTransition(init_matrix, betas)

def get_transition(transition_type):
    transition = {
        'none': NoTransition,
        'categorical': categorical_transition,
        'dirichlet': dirichlet_transition
    }[transition_type]()
    return transition

def get_regularization(regularization_type, num_pairs, gamma):
    regularization = {
        'none': no_regularization,
        'tv': tv_regularization(num_pairs),
    }[regularization_type]
    return regularization, gamma

def get_train_step(model, transition, optimizer, regularization=None, gamma=0.):
    if regularization is None or gamma == 0.:
        regularization, gamma = no_regularization, 0.
    device = next(model.parameters()).device
    def step(x, y):
        model.train()
        # device
        x = x.to(device)
        y = y.to(device)
        # forward
        t = model(x)
        l = transition.loss(t, y) - gamma * regularization(t)
        # backward
        optimizer.zero_grad()
        l.backward()
        # optimization
        optimizer.step()  # optimize model
        transition.update(t, y)  # optimize transition

    return step

def predict_test(model: nn.Module, loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    ts = []
    ys = []
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for x, y,_ in loader:
            ts.append(model(x.to(device)))
            ys.append(y.to(device))
    t = torch.cat(ts)
    y = torch.cat(ys)
    return t, y

def predict_train(model: nn.Module, loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    ts = []
    ys = []
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for x, y,_,_ in loader:
            ts.append(model(x.to(device)))
            ys.append(y.to(device))
    t = torch.cat(ts)
    y = torch.cat(ys)
    return t, y

def take_cycle(n: int, xs: Iterable):
    # take n (cycle xs)
    it = iter(xs)
    for i in range(n):
        try:
            yield next(it)
        except StopIteration:
            it = iter(xs)
            yield next(it)