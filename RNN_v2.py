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
                 kernel_size, lstm_hidden, use_gpu, batch_size, n_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size # Should just be 1.
        self.num_filters = num_filters
        self.output_size = output_size # Number of AAs
        self.n_layers = n_layers # Defaults to one.

        self.kernel_size = kernel_size
        self.lstm_hidden = lstm_hidden
        self.use_gpu = use_gpu
        self.batch_size = batch_size

        self.convs = []
        self.conv_outputs = 0

        # Assuming kernel size is a list of lists. We make Sequential
        # convolutional elements for things in the same list. Later lists
        # are parallel convolutional layers.
        for i in xrange(1):
            inp_size = self.input_size
            mods = []
            for j in xrange(len(kernel_size)):
                kernel = kernel_size[j]
                nf = self.num_filters[j]
                # We want a conv, batchnorm and relu after each layer.
                mods.append(nn.Conv1d(inp_size, nf, kernel))
                mods.append(nn.BatchNorm1d(nf))
                mods.append(nn.ReLU())
                inp_size = nf
            # This is the total number of inputs to the LSTM layer.
            self.conv_outputs += nf
            self.convs.append(nn.Sequential(*mods))

        self.lstm_in_size = self.conv_outputs * 2
        self.convs = nn.ModuleList(self.convs)
        self.lstm = nn.LSTM(self.lstm_in_size, lstm_hidden, n_layers, dropout=0.01)
        self.out = nn.Linear(lstm_hidden, output_size)
        self.hidden = self.__init_hidden()


    def forward(self, inputs, chars, hidden):

        inputs = inputs.transpose(0, 1)
        # Run through Convolutional layers. Chomp elements so our output
        # size matches our labels. We basically want to ignore all the
        # elements that are convolving over the padding to the right of the
        # chars.

        # Originally had separate convolutional layers for each.
        # but not anymore.
        outs = [self.convs[0](inputs[n, :, :].unsqueeze(-2)) for n in xrange(inputs.size(0))]
        # print(outs[0].size())
        # len_to_add = chars.size(1) -
        for i in range(len(outs)):
            to_add = np.full((outs[i].size(0), outs[i].size(1)), -2 + i)
            to_add = add_cuda_to_variable(to_add, self.use_gpu).unsqueeze(-1)
            # print(to_add.size())
            outs[i] = torch.cat([to_add, outs[i]], 2)
            # print(outs[i].size())
        # print(chars.size())
        c = torch.cat([out for out in outs], 1)
        # print(c.size())

        # Turn (batch_size x hidden_size x seq_len) back into (seq_len x batch_size x hidden_size) for RNN
        p = c.transpose(1, 2).transpose(0, 1)
        # Repeat it so that it matches the expected input of the network.
        chars = chars.transpose(0, 1).unsqueeze(-1).repeat(1, 1, self.lstm_in_size)

        _, self.hidden = self.lstm(p, hidden)
        output, self.hidden = self.lstm(chars, self.hidden)

        conv_seq_len = output.size(0)
        output = self.out(F.relu(output))
        output = output.view(conv_seq_len, -1, self.output_size)
        return F.log_softmax(output)

    def __init_hidden(self):
            # The axes semantics are (num_layers, minibatch_size, hidden_dim)
            if self.use_gpu:
                self.hidden = (Variable(torch.zeros(1, self.batch_size, self.lstm_hidden).cuda()),
                               Variable(torch.zeros(1, self.batch_size, self.lstm_hidden).cuda()))
            else:
                self.hidden = (Variable(torch.zeros(1, self.batch_size, self.lstm_hidden)),
                               Variable(torch.zeros(1, self.batch_size, self.lstm_hidden)))

    def init_hidden():
        self.__init_hidden()

    def train(self,
              fasta_sampler,
              batch_size,
              epochs,
              lr,
              samples_per_epoch=100000,
              save_params=None,
              slice_len=200,
              slice_incr_perc=0.1):
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
                min2, min1, min0, targets = fasta_sampler.generate_N_random_samples_and_targets(self.batch_size)

                min2 = add_cuda_to_variable(min2, self.use_gpu)
                min1 = add_cuda_to_variable(min1, self.use_gpu)
                min0 = add_cuda_to_variable(min0, self.use_gpu)
                targets = add_cuda_to_variable(targets, self.use_gpu)
                train = torch.stack([min2, min1], 1)

                self.zero_grad()
                self.__init_hidden()
                loss = 0

                # Do a forward pass.
                outputs = self.forward(train, min0, self.hidden)
                targets = targets.long().transpose(0,1).unsqueeze(-1).long()


                for bat in range(batch_size):
                    loss += loss_function(outputs[:, bat, :], targets[:, bat, :].squeeze(1))
                loss.backward()
                optimizer.step()

                if iterate % 1000 == 0:
                    print('Loss ' + str(loss.data[0] / self.batch_size))
                    min2, min1, min0, targets = fasta_sampler.generate_N_random_samples_and_targets(self.batch_size, group='validation')

                    min2 = add_cuda_to_variable(min2, self.use_gpu)
                    min1 = add_cuda_to_variable(min1, self.use_gpu)
                    min0 = add_cuda_to_variable(min0, self.use_gpu)
                    targets = add_cuda_to_variable(targets, self.use_gpu)
                    train = torch.stack([min2, min1], 1)

                    self.__init_hidden()
                    outputs_val = self.forward(train, min0, self.hidden)
                    outputs_val = outputs_val
                    targets = targets.long().transpose(0,1).unsqueeze(-1).long()
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
                slice_len = int(slice_len)
                print('Increased slice length to: {}'.format(slice_len))

            if save_params is not None:
                torch.save(self.state_dict(), save_params[0])
                with open(save_params[1], 'w+') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                    writer.writerow(train_loss_vec)
                    writer.writerow(val_loss_vec)
                print('Saved model state to: {}'.format(save_params[0]))


        return train_loss_vec, val_loss_vec

    def daydream(self, primer, T, fasta_sampler, predict_len=None):
        vocab_size = len(fasta_sampler.vocabulary)
        # Have we detected an end character?
        end_found = False
        self.batch_size = 1

        self.__init_hidden()
        primer_input = [fasta_sampler.vocabulary[char] for char in primer]

        self.seq_len = len(primer_input)
        # build hidden layer
        inp = add_cuda_to_variable(primer_input, self.use_gpu).unsqueeze(-1).transpose(0, 2)
        _ = self.forward(inp, self.hidden)

        # self.seq_len = 1
        predicted = list(primer_input)
        if predict_len is not None:
            for p in range(predict_len):
                inp = add_cuda_to_variable(predicted, self.use_gpu).unsqueeze(-1).transpose(0, 2)
                output = self.forward(inp, self.hidden)[-1]
                soft_out = custom_softmax(output.data.squeeze(), T)
                found_char = flip_coin(soft_out, self.use_gpu)
                predicted.append(found_char)

        else:
            while end_found is False:
                inp = add_cuda_to_variable(predicted, self.use_gpu).unsqueeze(-1).transpose(0, 2)
                output = self.forward(inp, self.hidden)[-1]
                soft_out = custom_softmax(output.data.squeeze(), T)
                found_char = flip_coin(soft_out, self.use_gpu)
                predicted.append(found_char)
                if found_char == fasta_sampler.vocabulary[fasta_sampler.end]:
                    end_found = True

        strlist = [fasta_sampler.inverse_vocabulary[pred] for pred in predicted]
        return ''.join(strlist)
        # return (''.join(strlist).replace(fasta_sampler.pad_char, '')).replace(fasta_sampler.start, '').replace(fasta_sampler.end, '')
