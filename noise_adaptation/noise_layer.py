import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


class NoiseLayer(nn.Module):
    def __init__(self, theta, hidden, num_classes = 10):
        super(NoiseLayer, self).__init__()
        u = Variable(torch.randn(num_classes,num_classes,hidden)/100,requires_grad=True)
        self.u = torch.nn.Parameter(data=u, requires_grad=True)
        b = Variable(theta,requires_grad=True)
        self.b = torch.nn.Parameter(data=b, requires_grad=True)
        # self.init_param()

    def forward(self, h):
        z = torch.matmul(self.u,h.T).T+self.b
        z = torch.softmax(z,dim=-1)
        return z


if __name__ == '__main__':
    h = torch.randn(1,128)
    theta = torch.eye(10)
    net = NoiseLayer(theta,128)
    z = net(h)
    print(z.shape)
    print(net.parameters())