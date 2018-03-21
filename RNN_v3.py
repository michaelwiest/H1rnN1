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
                 kernel_size, dilation, lstm_hidden, use_gpu, batch_size, n_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size # Should just be 1.
        self.num_filters = num_filters
        self.output_size = output_size # Number of AAs
        self.n_layers = n_layers # Defaults to one.

        self.kernel_size = kernel_size
        self.dilation = dilation
        self.lstm_hidden = lstm_hidden
        self.use_gpu = use_gpu
        self.batch_size = batch_size

        self.convs = []
        self.conv_outputs = 0

        # Assuming kernel size is a list of lists. We make Sequential
        # convolutional elements for things in the same list. Later lists
        # are parallel convolutional layers.

        for i in xrange(len(kernel_size)):
            inp_size = self.input_size
            mods = []
            k_row = kernel_size[i]
            for j in xrange(len(k_row)):
                kernel = k_row[j]
                nf = self.num_filters[i][j]
                pad = kernel
                # We want a conv, batchnorm and relu after each layer.
                mods.append(nn.Conv1d(inp_size, nf, kernel, padding=pad))
                mods.append(nn.BatchNorm1d(nf))
                mods.append(nn.ReLU())
                inp_size = nf

            self.conv_outputs += nf
            self.convs.append(nn.Sequential(*mods))

        # This is the total number of inputs to the LSTM layer.
        self.lstm_in_size = 2 * (self.conv_outputs + self.input_size) # 2x prev years
        self.convs = nn.ModuleList(self.convs)
        self.lstm = nn.LSTM(self.lstm_in_size, lstm_hidden, n_layers, dropout=0.01)
        self.out = nn.Linear(lstm_hidden, output_size)
        self.hidden = self.__init_hidden()

    def forward(self, inputs, hidden):

        # inputs = [min2, min1] years
        batch_size = inputs.size(2)
        # The number of characters in the input string
        num_elements = inputs.size(3)

        # Run through Convolutional layers. Chomp elements at the end because they correspons
        # to convolution with right-side padding, which have "no valuable" information explicitely.
        # This way we get matching sizes of conv sequences to input sequences and can concatenate.

        c = inputs[0].permute(1,0,2)
        if self.convs:
            inputs2 = inputs[:,0,:,:].permute(1,0,2)
            inputs1 = inputs[:,1,:,:].permute(1,0,2)

            outs2 = [conv(inputs2)[:, :, :num_elements] for conv in self.convs] #2 prev years
            outs1 = [conv(inputs1)[:, :, :num_elements] for conv in self.convs] #1 prev years
            c2 = torch.cat([out for out in outs2], 1)
            c1 = torch.cat([out for out in outs1], 1)
            c = torch.cat([c, c2, c1, ],1) # Append raw sequences

        # Turn (batch_size x hidden_size x seq_len) back into (seq_len x batch_size x hidden_size) for RNN
        p = c.permute(2, 0, 1)

        output, self.hidden = self.lstm(p, hidden)
        conv_seq_len = output.size(0)
        output = self.out(F.relu(output))
        output = output.view(conv_seq_len, -1, self.output_size)
        return output
        # return F.log_softmax(output)

    def __init_hidden(self):
            # The axes semantics are (num_layers, minibatch_size, hidden_dim)
            if self.use_gpu:
                self.hidden = (Variable(torch.zeros(1, self.batch_size, self.lstm_hidden).cuda()),
                               Variable(torch.zeros(1, self.batch_size, self.lstm_hidden).cuda()))
            else:
                self.hidden = (Variable(torch.zeros(1, self.batch_size, self.lstm_hidden)),
                               Variable(torch.zeros(1, self.batch_size, self.lstm_hidden)))

    def init_hidden(self):
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

                train, targets, _ = fasta_sampler.generate_N_random_samples_and_targets(self.batch_size, slice_len=slice_len)
                train = add_cuda_to_variable(train, self.use_gpu)
                targets = add_cuda_to_variable(targets, self.use_gpu)
                # Train = [min2, min1] 2 years ago, targets = current year

                self.zero_grad()
                self.__init_hidden()
                loss = 0

                # Do a forward pass.
                outputs = self.forward(train, self.hidden)
                # print(outputs.size())

                # Need to skip the first entry in the predicted elements.
                # and also ignore all the end elements because theyre just
                # predicting padding.
                # outputs = outputs[1:-self.kernel_size, :, :]
                # reshape the targets to match.

                targets = targets.permute(1,0).unsqueeze(2).long()

                for bat in range(batch_size):
                    loss += loss_function(outputs[:, bat, :], targets[:, bat, :].squeeze(1))
                loss.backward()
                optimizer.step()

                if iterate % 1000 == 0:
                    print('Loss ' + str(loss.data[0] / self.batch_size))
                    val, val_targets, _ = fasta_sampler.generate_N_random_samples_and_targets(self.batch_size,
                                                                                          group='validation',
                                                                                          slice_len=slice_len)
                    val = add_cuda_to_variable(val, self.use_gpu)
                    val_targets = add_cuda_to_variable(val_targets, self.use_gpu)

                    self.__init_hidden()
                    outputs_val = self.forward(val, self.hidden)
                    outputs_val = outputs_val
                    val_targets = val_targets.permute(1,0).unsqueeze(2).long()
                    # val_targets = val_targets.transpose(0, 2).transpose(1, 2).long()
                    val_loss = 0
                    for bat in range(self.batch_size):
                        val_loss += loss_function(outputs_val[:, 1, :], val_targets[:, 1, :].squeeze(1))
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

    def daydream1(self, primer, T, fasta_sampler, predict_len=None):

        # ex = fs.generate_N_sample_per_year(1, 2012, full=False, to_num=False)[0]
        # train, targets, _ = fasta_sampler.generate_N_random_samples_and_targets(self.batch_size, slice_len=slice_len)

        prev_year2 = primer[0][:]
        prev_year1 = primer[1][:]
        current_year = primer[2][:]

        vocab_size = len(fasta_sampler.vocabulary)
        # Have we detected an end character?
        end_found = False
        self.batch_size = 1

        primer_input2 = [fasta_sampler.vocabulary[char] for char in prev_year2]
        primer_input1 = [fasta_sampler.vocabulary[char] for char in prev_year1]
        primer_input = [primer_input2,primer_input1]

        primer_input = add_cuda_to_variable(primer_input, self.use_gpu).unsqueeze(2)

        # current_output = [fasta_sampler.vocabulary[char] for char in current_year]
        # current_output = add_cuda_to_variable(current_output, self.use_gpu)

        self.init_hidden()
        self.zero_grad()
        loss = 0

        # Do a forward pass.
        predictions = self.forward(primer_input, self.hidden)
        predictions = predictions.data.squeeze()


        print(predictions.size(1))

        soft_out = custom_softmax2(predictions, T)
        print(soft_out.size(1))

        pred_seq = np.argmax(soft_out, axis=1)

        strlist = [fasta_sampler.inverse_vocabulary[pred] for pred in pred_seq]

        # found_char = flip_coin(soft_out, self.use_gpu)


        # # self.seq_len = 1
        # predicted = list(primer_input)
        # if predict_len is not None:
        #     for p in range(predict_len):
        #         inp = add_cuda_to_variable(predicted, self.use_gpu).unsqueeze(-1).transpose(0, 2)
        #         output = self.forward(inp, self.hidden)[-1]
        #         soft_out = custom_softmax(output.data.squeeze(), T)
        #         found_char = flip_coin(soft_out, self.use_gpu)
        #         predicted.append(found_char)
        #
        # else:
        #     while end_found is False:
        #         inp = add_cuda_to_variable(predicted, self.use_gpu).unsqueeze(-1).transpose(0, 2)
        #         output = self.forward(inp, self.hidden)[-1]
        #         soft_out = custom_softmax(output.data.squeeze(), T)
        #         found_char = flip_coin(soft_out, self.use_gpu)
        #         predicted.append(found_char)
        #         if found_char == fasta_sampler.vocabulary[fasta_sampler.end]:
        #             end_found = True

        # strlist = [fasta_sampler.inverse_vocabulary[pred] for pred in predicted]
        return ''.join(strlist)
        # return (''.join(strlist).replace(fasta_sampler.pad_char, '')).replace(fasta_sampler.start, '').replace(fasta_sampler.end, '')
