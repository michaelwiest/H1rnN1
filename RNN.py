from __future__ import print_function
import torch.autograd as autograd
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random
import pdb
import numpy as np
from helper import *

class RNN(nn.Module):
    def __init__(self, input_size, num_filters, output_size,
                 kernel_size, lstm_hidden, use_gpu, batch_size, n_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size # Should just be 1.
        self.num_filters = num_filters
        self.output_size = output_size # Number of AAs
        self.n_layers = n_layers # Defaults to one.

        if kernel_size % 2 != 0:
            raise ValueError('Please supply an even number for kernel size')
        self.kernel_size = kernel_size
        self.lstm_hidden = lstm_hidden
        self.use_gpu = use_gpu
        self.batch_size = batch_size

        self.c1 = nn.Conv1d(input_size, num_filters, kernel_size)
        # self.c2 = nn.Conv1d(hidden_size, hidden_size, 1)
        self.lstm = nn.LSTM(num_filters, lstm_hidden, n_layers, dropout=0.01)
        self.out = nn.Linear(lstm_hidden, output_size)
        self.hidden = self.__init_hidden()

    def forward(self, inputs, hidden):
        batch_size = inputs.size(1)

        # Need this here to get the datatypes to work for some reason.
        # inputs = add_cuda_to_variable(Variable(inputs.data.type(torch.FloatTensor)), self.use_gpu)
        # Turn (seq_len x batch_size x input_size) into (batch_size x input_size x seq_len) for CNN
        # inputs = inputs.transpose(0, 1).transpose(1, 2)

        # Run through Conv1d and Pool1d layers
        c = self.c1(inputs)
        # c = self.c2(p)

        # Turn (batch_size x hidden_size x seq_len) back into (seq_len x batch_size x hidden_size) for RNN
        p = c.transpose(1, 2).transpose(0, 1)

        # p = F(p)
        output, self.hidden = self.lstm(p, hidden)
        conv_seq_len = output.size(0)
        # output = output.view(conv_seq_len * batch_size, -1) # Treating (conv_seq_len x batch_size) as batch_size for linear layer
        output = self.out(F.relu(output))
        output = output.view(conv_seq_len, -1, self.output_size)
        return output


    def __init_hidden(self):
            # The axes semantics are (num_layers, minibatch_size, hidden_dim)
            if self.use_gpu:
                self.hidden = (Variable(torch.zeros(1, self.batch_size, self.lstm_hidden).cuda()),
                        Variable(torch.zeros(1, self.batch_size, self.lstm_hidden).cuda()))
            else:
                self.hidden =  (Variable(torch.zeros(1, self.batch_size, self.lstm_hidden)),
                        Variable(torch.zeros(1, self.batch_size, self.lstm_hidden)))

    def init_hidden():
        self.__init_hidden()


    def train(self, fasta_sampler, batch_size, epochs, lr, samples_per_epoch=100000):
        np.random.seed(1)

        self.batch_size = batch_size

        print(self.use_gpu)

        if self.use_gpu:
            print('calling cuda on self.')
            self.cuda()

        loss_function = nn.CrossEntropyLoss()
        # Try Adagrad & RMSProp
        optimizer = optim.SGD(self.parameters(), lr=lr)

        # For logging the data for plotting
        train_loss_vec = []
        val_loss_vec = []

        for epoch in range(epochs):

            '''
            Visit each possible example once. Can maybe tweak this to be more
            stochastic.
            '''
            for iterate in range(int(samples_per_epoch / self.batch_size)):
                train, targets = fasta_sampler.generate_N_random_samples_and_targets(self.batch_size, self.kernel_size)
                train = add_cuda_to_variable(train, self.use_gpu)
                print(train)
                targets = add_cuda_to_variable(targets, self.use_gpu)
                self.zero_grad()
                self.__init_hidden()
                outputs = self.forward(train, self.hidden)

                loss = 0
                # print(outputs[1:, :, :].shape)
                # print(targets.transpose(0,2).transpose(1,2).shape)
                for bat in range(batch_size):
                    loss += loss_function(outputs[1:, bat, :], targets[:, bat, :].squeeze(1))
                loss.backward()
                optimizer.step()

                if iterate % 2000 == 0:
                    print('Loss ' + str(loss.data[0] / self.batch_size))
                    val_indices = random.sample(possible_val_indices, self.batch_size)
                    val_inputs, val_targets = self.__convert_examples_to_targets_and_slices(val_data, val_indices, seq_len, ex_idx)

                    val_inputs = add_cuda_to_variable(val_inputs, self.use_gpu)
                    val_targets = add_cuda_to_variable(val_targets, self.use_gpu)
                    self.__init_hidden()
                    outputs_val = self.__forward(val_inputs)
                    val_loss = 0
                    for bat in range(self.batch_size):
                        val_loss += loss_function(outputs_val[:,1,:], val_targets[:,1,:].squeeze(1))
                    val_loss_vec.append(val_loss.data[0] / self.batch_size)
                    train_loss_vec.append(loss.data[0] / self.batch_size)
                    print('Validataion Loss ' + str(val_loss.data[0]/batch_size))
                iterate += 1
            print('Completed Epoch ' + str(epoch))

        return train_loss_vec, val_loss_vec
