from fasta_sampler import *
from RNN import *
from helper import *
import csv
import IPython

batch_size = 10
kernel_size = 4
lstm_hidden_units = 100
num_filters = 32
samples_per_epoch = 50000
num_epochs = 10
learning_rate = 0.001

# Build the data handler object.
fs = FastaSampler('data/HA_n_2010_2018.fa', 'data/HA_s_2010_2018.fa')
# Assign the validation years.
fs.set_train_val_years([2016, 2017])
vocab = fs.vocabulary


use_gpu = torch.cuda.is_available()

rnn = RNN(1, num_filters, len(vocab.keys()), kernel_size, lstm_hidden_units,
          use_gpu, batch_size)

train_loss, val_loss = rnn.train(fs, batch_size,
                                 num_epochs,
                                 learning_rate,
                                 samples_per_epoch=samples_per_epoch)

torch.save(rnn.state_dict(), 'model.pt')

with open('log.csv', 'w+') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(train_loss)
    writer.writerow(val_loss)
