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
from helper_v2 import *
import csv
import IPython

class RNN(nn.Module):
    def __init__(self, input_size, num_filters, output_size,
                 kernel_size, use_gpu, batch_size, n_layers=1,
                 unique_convs=False,
                 num_aas=568,
                 pool=False):
        super(RNN, self).__init__()
        self.input_size = input_size # Should just be 1.
        self.num_filters = num_filters
        self.output_size = output_size # Number of AAs
        self.n_layers = n_layers # Defaults to one.
        self.num_aas = num_aas

        self.kernel_size = kernel_size
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.unique_convs = unique_convs

        self.convs = []
        self.conv_outputs = 0
        # How many time steps back to look.
        self.num_previous_sequences = 2

        # Construct a convolutional network for each input of preceeding
        # AA sequences.
        if self.unique_convs:
            num_convs = self.num_previous_sequences
        else:
            num_convs = 1
        for i in xrange(num_convs):
            inp_size = self.input_size
            mods = []
            for j in xrange(len(kernel_size)):
                kernel = kernel_size[j]
                nf = self.num_filters[j]
                # We want a conv, batchnorm and relu after each layer.
                mods.append(nn.Conv1d(inp_size, nf, kernel))
                mods.append(nn.BatchNorm1d(nf))
                mods.append(nn.ReLU())
                mods.append(nn.Dropout2d())
                if pool:
                    mods.append(nn.MaxPool1d(2, stride=2))
                inp_size = nf
            # This is the total number of inputs to the LSTM layer.
            self.conv_outputs += nf
            self.convs.append(nn.Sequential(*mods))
        # THis is hard coded right now but is a function of the kernels

        self.conv_size = self.num_aas - sum([k - 1 for k in kernel_size])

        self.num_conv_filters = self.conv_outputs * self.num_previous_sequences
        self.convs = nn.ModuleList(self.convs)
        self.lstm = nn.LSTM(1, self.conv_size + 1, 1, dropout=0.15)
        self.lin0 = nn.Linear(self.conv_size + 1, self.conv_size + 1)
        self.lin1 = nn.Linear(self.conv_size + 1, output_size)
        self.lin2 = nn.Linear(self.num_conv_filters, 1)
        self.tanh = nn.Tanh()
        self.hidden = None


    def forward(self,
                inputs,
                aa_string,
                reset_hidden=True
                ):

        inputs = inputs.transpose(0, 1)

        # If we have separate convolutional networks for each previous input
        # then use those, otherwise just use one network.
        if self.unique_convs:
            outs = [self.convs[n](inputs[n, :, :].unsqueeze(-2)) for n in xrange(inputs.size(0))]
        else:
            outs = [self.convs[0](inputs[n, :, :].unsqueeze(-2)) for n in xrange(inputs.size(0))]
        # Prefix each of the outputs of the convolution with a digit representing
        # how far back in time they are (either -2 or -1)
        for i in range(len(outs)):
            to_add = np.full((outs[i].size(0), outs[i].size(1)),
                              -self.num_previous_sequences + i)
            to_add = add_cuda_to_variable(to_add, self.use_gpu).unsqueeze(-1)
            outs[i] = torch.cat([to_add, outs[i]], 2)

        conv_output = torch.cat([out for out in outs], 1)

        # Turn (batch_size x hidden_size x seq_len) back into (seq_len x batch_size x hidden_size) for RNN
        conv_output = conv_output.transpose(0, 1)
        # If we haven't set the hidden state yet. Basically we call this when
        # the model is trained and we want to seed it.
        # if self.hidden is None:
        if reset_hidden:
            self._set_hiden_to_conv(self.lin2(conv_output.transpose(0, 2)).transpose(0, 2))

        aa_string = aa_string.transpose(0, 1).unsqueeze(-1)
        output, self.hidden = self.lstm(aa_string, self.hidden)
        conv_seq_len = output.size(0)
        # output = self.lin0(self.tanh(output))
        output = self.lin1(self.tanh(output))
        output = output.view(conv_seq_len, -1, self.output_size)
        return F.log_softmax(output)

    def _set_hiden_to_conv(self, conv):
            self.hidden = (conv.contiguous(),
                           conv.contiguous())


    def __init_hidden(self):
            # The axes semantics are (num_layers, minibatch_size, hidden_dim)
            # Add one to conv size because they're prefixed with distance vals.
            if self.use_gpu:
                self.hidden = (Variable(torch.zeros(1, self.batch_size, self.conv_size + 1).cuda()),
                               Variable(torch.zeros(1, self.batch_size, self.conv_size + 1).cuda()))
            else:
                self.hidden = (Variable(torch.zeros(1, self.batch_size, self.conv_size + 1)),
                               Variable(torch.zeros(1, self.batch_size, self.conv_size + 1)))


    def train(self,
              fasta_sampler,
              batch_size,
              epochs,
              lr,
              samples_per_epoch=100000,
              save_params=None,
              slice_len=200,
              slice_incr_perc=0.1
              ):
        np.random.seed(1)

        self.batch_size = batch_size

        if self.use_gpu:
            self.cuda()

        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=lr)

        # For logging the data for plotting
        train_loss_vec = []
        val_loss_vec = []

        for epoch in range(epochs):

            '''
            The number of steps to do between epochs pretty arbitrary.
            '''
            for iterate in range(int(samples_per_epoch / self.batch_size)):
                # Get the samples and make them cuda.
                prevs, current, targets = fasta_sampler.generate_N_random_samples_and_targets(self.batch_size, group='validation',
                                                                                              slice_len=slice_len)
                prevs = [add_cuda_to_variable(p, self.use_gpu) for p in prevs]
                current = add_cuda_to_variable(current, self.use_gpu)
                targets = add_cuda_to_variable(targets, self.use_gpu)
                train = torch.stack(prevs, 1)

                self.zero_grad()
                self.__init_hidden()

                loss = 0

                # Do a forward pass.
                outputs = self.forward(train, current)
                targets = targets.long().transpose(0, 1).unsqueeze(-1).long()


                for bat in range(batch_size):
                    loss += loss_function(outputs[:, bat, :], targets[:, bat, :].squeeze(1))
                loss.backward()
                optimizer.step()

                if iterate % 1000 == 0:
                    print('Loss ' + str(loss.data[0] / self.batch_size))
                    prevs, current, targets = fasta_sampler.generate_N_random_samples_and_targets(self.batch_size, group='validation',
                                                                                                  slice_len=slice_len)
                    prevs = [add_cuda_to_variable(p, self.use_gpu) for p in prevs]
                    current = add_cuda_to_variable(current, self.use_gpu)
                    targets = add_cuda_to_variable(targets, self.use_gpu)
                    train = torch.stack(prevs, 1)

                    self.__init_hidden()
                    outputs_val = self.forward(train, current)
                    outputs_val = outputs_val
                    targets = targets.long().transpose(0, 1).unsqueeze(-1).long()
                    val_loss = 0
                    for bat in range(self.batch_size):
                        val_loss += loss_function(outputs_val[:, bat, :], targets[:, bat, :].squeeze(1))
                    val_loss_vec.append(val_loss.data[0] / self.batch_size)
                    train_loss_vec.append(loss.data[0] / self.batch_size)
                    print('Validataion Loss ' + str(val_loss.data[0]/batch_size))
                iterate += 1
            print('Completed Epoch ' + str(epoch))

            if slice_incr_perc is not None:
                slice_len += slice_len * slice_incr_perc
                slice_len = min(self.num_aas - 1, int(slice_len))
                print('Increased slice length to: {}'.format(slice_len))

            if save_params is not None:
                torch.save(self.state_dict(), save_params[0])
                with open(save_params[1], 'w+') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                    writer.writerow(train_loss_vec)
                    writer.writerow(val_loss_vec)
                print('Saved model state to: {}'.format(save_params[0]))


        return train_loss_vec, val_loss_vec

    def daydream(self, primer, prev_observations, T, fasta_sampler, predict_len):
        vocab_size = len(fasta_sampler.vocabulary)
        # Have we detected an end character?
        self.batch_size = 1

        self.__init_hidden()
        prev_observations = [add_cuda_to_variable(o, self.use_gpu) for o in prev_observations]
        train = torch.stack(prev_observations, 1)

        self.seq_len = len(primer)
        # build hidden layer
        # inp = add_cuda_to_variable(primer[:-1], self.use_gpu)
        # _ = self.forward(train, inp)

        # self.seq_len = 1
        predicted = list(primer)
        if predict_len is not None:
            for p in range(predict_len):
                inp = add_cuda_to_variable(predicted, self.use_gpu)
                output = self.forward(train, inp, reset_hidden=False)[-1]
                soft_out = custom_softmax(output.data.squeeze(), T)
                found_char = flip_coin(soft_out, self.use_gpu)
                predicted.append(found_char)

        strlist = [fasta_sampler.inverse_vocabulary[pred] for pred in predicted]
        return ''.join(strlist)
