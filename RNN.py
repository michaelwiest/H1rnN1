import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

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
        inputs = Variable(inputs.data.type(torch.FloatTensor))
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
        ex_size = len(examples)
        np.random.seed(1)

        self.batch_size = batch_size

        if self.use_gpu:
            self.cuda()

        loss_function = nn.CrossEntropyLoss()
        # Try Adagrad & RMSProp
        optimizer = optim.SGD(self.parameters(), lr=lr)

        # For logging the data for plotting
        train_loss_vec = []
        val_loss_vec = []

        for epoch in range(epochs):
            #get random slice
            # possible_example_indices = range(len(training_data))             #Idx of training examples
            # possible_slice_starts = [range(len(ex)) for ex in training_data] #len of each example
            # possible_val_indices = range(len(val_data))                      #Idx of val examples
            # after going through all of a , will have gone through all possible 30
            # character slices
            # iterate = 0

            '''
            Visit each possible example once. Can maybe tweak this to be more
            stochastic.
            '''
            for iterate in range(int(samples_per_epoch / self.batch_size)):
                train, targets = fasta_sampler.generate_N_random_samples_and_targets(self.batch_size, self.kernel_size)
                train = add_cuda_to_variable(train, use_gpu)
                targets = add_cuda_to_variable(targets, use_gpu)
                self.zero_grad()
                self.__init_hidden()
                outputs = self.__forward(train)

                loss = 0
                print(outputs.shape)
                print(targets.shape)
                for bat in range(batch_size):
                    loss += loss_function(outputs[:,bat,:], targets[:,bat,:].squeeze(1))
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

        # return train_loss_vec, val_loss_vec
        #     while len(possible_example_indices) > self.batch_size:
        #         #Get #(batch_size) random training examples to take samples from
        #         example_indices = random.sample(possible_example_indices, self.batch_size)
        #
        #         # Get processed data.
        #         # print(len(possible_slice_starts[example_indices[0]]))
        #         len_old = len(possible_example_indices)
        #
        #         rand_slice, targets = self.__convert_examples_to_targets_and_slices(training_data,
        #                                                                             example_indices,
        #                                                                             seq_len, ex_idx,
        #                                                                             center=False,
        #                                                                             possible_slice_starts=possible_slice_starts,
        #                                                                             possible_example_indices=possible_example_indices)
        #
        #         # print(len(possible_slice_starts[example_indices[0]]))
        #         # if len_old != len(possible_example_indices):
        #         #     print(len(possible_example_indices))
        #         #     print('---')
        #         # prepare data and targets for self
        #         rand_slice = add_cuda_to_variable(rand_slice, self.use_gpu)
        #         targets = add_cuda_to_variable(targets, self.use_gpu)
        #
        #         # Pytorch accumulates gradients. We need to clear them out before each instance
        #         self.zero_grad()
        #
        #         # Also, we need to clear out the hidden state of the LSTM,
        #         # detaching it from its history on the last instance.
        #         self.__init_hidden()
        #         # From TA:
        #         # another option is to feed sequences sequentially and let hidden state continue
        #         # could feed whole sequence, and then would kill hidden state
        #
        #         # Run our __forward pass.
        #
        #         outputs = self.__forward(rand_slice)
        #         # Step 4. Compute the loss, gradients, and update the parameters by
        #         #  calling optimizer.step()
        #         loss = 0
        #         for bat in range(batch_size):
        #             loss += loss_function(outputs[:,bat,:], targets[:,bat,:].squeeze(1))
        #         loss.backward()
        #         optimizer.step()
        #
        #         if iterate % 2000 == 0:
        #             print('Loss ' + str(loss.data[0] / self.batch_size))
        #             val_indices = random.sample(possible_val_indices, self.batch_size)
        #             val_inputs, val_targets = self.__convert_examples_to_targets_and_slices(val_data, val_indices, seq_len, ex_idx)
        #
        #             val_inputs = add_cuda_to_variable(val_inputs, self.use_gpu)
        #             val_targets = add_cuda_to_variable(val_targets, self.use_gpu)
        #             self.__init_hidden()
        #             outputs_val = self.__forward(val_inputs)
        #             val_loss = 0
        #             for bat in range(self.batch_size):
        #                 val_loss += loss_function(outputs_val[:,1,:], val_targets[:,1,:].squeeze(1))
        #             val_loss_vec.append(val_loss.data[0] / self.batch_size)
        #             train_loss_vec.append(loss.data[0] / self.batch_size)
        #             print('Validataion Loss ' + str(val_loss.data[0]/batch_size))
        #         iterate += 1
        #     print('Completed Epoch ' + str(epoch))
        #
        # return train_loss_vec, val_loss_vec
