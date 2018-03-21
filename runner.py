from fasta_sampler import *
from fasta_sampler_v2 import *
from RNN_v2 import *
from helper import *
import csv
import numpy as np

# batch_size = 30
# kernel_sizes = [3, 5]
# num_filters = [16, 64]
# samples_per_epoch = 50000
# num_epochs = 15
# learning_rate = 0.003
# seq_len = 100
# slice_incr_perc = 0.1

batch_size = 30
kernel_sizes = [3, 5, 7]
num_filters = [16, 64, 64]
samples_per_epoch = 100000
num_epochs = 25
learning_rate = 0.002
seq_len = 100
slice_incr_perc = 0.1

# Build the data handler object.
fs = FastaSamplerV2('data/HA_n_2010_2018.fa', 'data/HA_s_2010_2018.fa', k = 10,
                   use_order=True)
# Assign the validation years.
fs.set_validation_years([2016, 2017])
vocab = fs.vocabulary

use_gpu = torch.cuda.is_available()

rnn = RNN(1,
          num_filters,
          len(vocab.keys()),
          kernel_sizes,
          use_gpu,
          batch_size
          )
# rnn.cuda(device_id=0)
rnn.load_state_dict(torch.load('random_fasta100_v3.pt', map_location=lambda storage, loc: storage))

# rnn = torch.load('random_fasta100_v2.pt', map_location=lambda storage, loc: storage)
# rnn.load_state_dict(torch.load('random_fasta100_v3.pt'))

from evaluator import *
#from visualize import *
#vis = Visualize(2015, fs, rnn)
#dist_mat = vis.distance_heatmap()
eva = Evaluator(fs, rnn)
dist_mat, _, _ = eva.gen_and_compare_year(100, '$M', 2015, 10)
plt.imshow(dist_mat, cmap='PiYG', interpolation='nearest')
plt.colorbar()
plt.style.use('fivethirtyeight')
plt.xlabel('2016 North Hemisphere Actual Proteins')
plt.ylabel('2016 North Hemisphere Predicted Proteins')
plt.title('Distances between proteins')
plt.show()
