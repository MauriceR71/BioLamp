import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from matplotlib import pyplot as plt

data_mean = 4
data_stddev = 1.25

(name, preprocess, d_input_func) = ("Only 4 moments", lambda data: get_moments(data), lambda x: 4)
print("Using data [%s]" % (name))


def get_distribution_sampler(mu, sigma):
    return lambda n: torch.Tensor(np.random.normal(mu, sigma, (1, n)))


def get_generator_input_sampler():
    return lambda m, n: torch.rand(m, n)


class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, f):
        super(Generator, self).__init__()
        # nn.Module.__init__(self)
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.f = f

    def forward(self, x):
        x = 



#
# def get_moments(data):
#     # returns the first four moments of the data
#     mean = data.mean()
#     diffs = data - mean
#     var = torch.mean(torch.pow(diffs, 2.0))
#     std = torch.pow(var, 0.5)
#     zscores = diffs / std
#     skews = torch.mean(torch.pow(zscores, 3.0))
#     kurtosis = torch.mean(torch.pow(zscores, 4.0))
#     final = torch.cat((mean.reshape(1,), std.reshape(1,), skews.reshape(1,), kurtosis.reshape(1,)))
#     return final


