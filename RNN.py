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
        output, hidden = self.lstm(p, hidden)
        conv_seq_len = output.size(0)
        # output = output.view(conv_seq_len * batch_size, -1) # Treating (conv_seq_len x batch_size) as batch_size for linear layer
        output = self.out(F.relu(output))
        output = output.view(conv_seq_len, -1, self.output_size)
        return output, hidden


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
