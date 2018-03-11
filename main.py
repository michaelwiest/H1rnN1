from fasta_sampler import *
from RNN import *
from helper import *



# with open('data/HA_n_2010_2018.fa', 'r') as d:
#     all_data = d.read()
bs = 10
fs = FastaSampler('data/HA_n_2010_2018.fa', 'data/HA_s_2010_2018.fa')
fs.set_train_val_years([2016, 2017])
ex = fs.generate_N_sample_per_year(bs, 2016)
vocab = fs.vocabulary

# for y, l in fs.north.items():
#     print(y)
#     seqs = [obj['seq'] for obj in l]
#     print(len(set(seqs)))
#     print(len((seqs)))
#     print('--')



use_gpu = torch.cuda.is_available()


# inp = torch.randn(10, 1, 1702)
# print(inp.size())
kernel_size = 30
# ex, t = fs.generate_N_random_samples_and_targets(2, padding=kernel_size)
# print(len(t[0]))
# ex = add_cuda_to_variable(ex, use_gpu)
# ex = ex.type(torch.FloatTensor)
rnn = RNN(1, 10, len(vocab.keys()), kernel_size, 100, use_gpu, bs)

# print(fs.generate_train_samples(2, 'validation'))
# o = rnn.forward(ex, rnn.hidden)
# print(o)
tl, vl = rnn.train(fs, 30, 2, 0.001)
