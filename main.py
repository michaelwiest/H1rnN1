from fasta_sampler import *
from RNN import *
from helper import *
import IPython


# with open('data/HA_n_2010_2018.fa', 'r') as d:
#     all_data = d.read()
bs = 10
fs = FastaSampler('data/HA_n_2010_2018.fa', 'data/HA_s_2010_2018.fa')
ex = fs.generate_N_sample(bs, 2013)
print(len(ex[0]))
vocab = fs.vocabulary


use_gpu = torch.cuda.is_available()
ex = add_cuda_to_variable(ex, use_gpu)[0]
# ex = ex.type(torch.FloatTensor)
# inp = torch.randn(10, 1, 1702)
# print(inp.size())

rnn = RNN(1, 10, len(vocab.keys()), 30, 100, use_gpu, bs)
rnn.forward(ex, rnn.hidden)
