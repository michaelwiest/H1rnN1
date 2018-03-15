from fasta_sampler import *
from RNN import *
from helper import *
import csv

batch_size = 30
# List of lists of kernel sizes. Kernels in same list are sequential
# Kernels in separate lists happen in parallel.
kernel_sizes = [[4, 8]]
# Filter sizes associated with kernels above. Will throw an error if they
# dont' match
num_filters = [[32, 32, 32]]
dilation = [0, 1, 0] # Deprecated
lstm_hidden_units = 120
# num_filters = 64
samples_per_epoch = 100000
num_epochs = 5
learning_rate = 0.001
seq_length = 200
seq_length_incr_perc = 0.15

# Build the data handler object.
fs = FastaSampler('data/HA_n_2010_2018.fa', 'data/HA_s_2010_2018.fa')
# Assign the validation years.
fs.set_validation_years([2016, 2017])
vocab = fs.vocabulary


use_gpu = torch.cuda.is_available()

rnn = RNN(1, num_filters, len(vocab.keys()), kernel_sizes, dilation, lstm_hidden_units,
          use_gpu, batch_size)

model_name = 'model.pt'
log_name = 'log.csv'
train_loss, val_loss = rnn.train(fs, batch_size,
                                 num_epochs,
                                 learning_rate,
                                 samples_per_epoch=samples_per_epoch,
                                 save_params=(model_name, log_name),
                                 slice_len=seq_length,
                                 slice_incr_perc=seq_length_incr_perc
                                 )
