from fasta_sampler import *
from fasta_sampler_v2 import *
from RNN_v2 import *
from helper import *
import csv
import numpy as np

batch_size = 20
# List of lists of kernel sizes. Kernels in same list are sequential
# Kernels in separate lists happen in parallel.
kernel_sizes = [3, 5, 7]
# Filter sizes associated with kernels above. Will throw an error if they
# dont' match
num_filters = [64, 64, 64]
lstm_hidden_units = 100
# num_filters = 64
samples_per_epoch = 50000
num_epochs = 5
learning_rate = 0.005
seq_length = 200
seq_length_incr_perc = 0.1

# Build the data handler object.
fs = FastaSamplerV2('data/HA_n_2010_2018.fa', 'data/HA_s_2010_2018.fa')
# Assign the validation years.
fs.set_validation_years([2016, 2017])
vocab = fs.vocabulary


use_gpu = torch.cuda.is_available()

a1 = np.ones((50, 566))
a2 = 2*np.ones((50, 566))
a3 = 3* np.ones((50, 566))

inp = np.stack((a1,a2,a3), axis = 0)

t = 4 * np.ones((50))


rnn = RNN(1, num_filters, len(vocab.keys()), kernel_sizes, lstm_hidden_units,
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
