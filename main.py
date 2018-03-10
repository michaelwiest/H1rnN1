'''
from fasta_sampler import *
from RNN import *
from helper import *


# with open('data/HA_n_2010_2018.fa', 'r') as d:
#     all_data = d.read()
bs = 10
fs = FastaSampler('data/HA_n_2010_2018.fa', 'data/HA_s_2010_2018.fa')
ex = fs.generate_N_sample(bs, 2013)
print(len(ex[0]))
vocab = fs.vocabulary


use_gpu = torch.cuda.is_available()
ex = add_cuda_to_variable(ex, use_gpu)[0]

rnn = RNN(1, 10, len(vocab.keys()), 30, 100, use_gpu, bs)
rnn.forward(ex, rnn.hidden)
'''

from __future__ import print_function
import torch.autograd as autograd
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
#import matplotlib.pyplot as plt
import random
import pdb
import numpy as np

from helper import *
from fasta_sampler import *
from RNN import *

fs = FastaSampler('data/HA_n_2010_2018.fa','data/HA_s_2010_2018.fa')

batch_size = 10

ex = fs.generate_N_sample(batch_size, 2013)
vocab = fs.vocabulary
input_size = 1

# check for GPU
use_gpu = torch.cuda.is_available()
seq_len = 30
hidden_layer_size = 100
filter_num = 10
kernal_size = 30

batch_size = 10
num_epochs = 10
lr = 0.01

model = RNN(input_size, filter_num, len(vocab.keys()), kernal_size, hidden_layer_size, use_gpu, batch_size)

train_loss, val_loss = model.train(fs, ex, seq_len, batch_size,num_epochs, lr)

#plt.plot(range(len(val_loss)), val_loss)
#plt.plot(range(len(train_loss)), train_loss)
#plt.show()
