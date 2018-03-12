import numpy as np
import random
from torch.autograd import Variable
import torch
import pdb

# function maps each word to an index
def get_idx(char_data):
    word_to_ix = {}
    count = 1
    for word in char_data:
        if word not in word_to_ix:
            word_to_ix[word] = count
            count += 1
    return word_to_ix

def avg_len_music_file(data, ch):
    start_idxs = [i for i, ltr in enumerate(data) if ltr == ch]
    avg_len = [ (start_idxs[i+1]) - start_idxs[i] for i in xrange( 0,len(start_idxs)-1 )]
    return int(np.mean(avg_len))

def add_cuda_to_variable(data_nums, is_gpu):
    tensor = torch.FloatTensor(data_nums)
    if isinstance(data_nums, list):
        tensor = tensor.unsqueeze_(0)
    tensor = tensor.unsqueeze_(2)
    if is_gpu:
        return Variable(tensor.cuda())[0]
    else:
        return Variable(tensor)[0]

# returns prediction based on probabilites
def flip_coin(probabilities, is_gpu):
    stacked_probs = np.cumsum(probabilities)
    rand_int = random.random()
    if is_gpu:
        sp = stacked_probs[0].cpu().numpy()
    else:
        sp = stacked_probs.numpy()
    dist = abs(sp - rand_int)
    return np.argmin(dist)

def init_hidden(model):
    # The axes semantics are (num_layers, minibatch_size, hidden_dim)
    if model.use_gpu:
        model.hidden = (Variable(torch.zeros(1, model.batch_size, model.hidden_dim).cuda()),
                Variable(torch.zeros(1, model.batch_size, model.hidden_dim).cuda()))
    else:
        model.hidden =  (Variable(torch.zeros(1, model.batch_size, model.hidden_dim)),
                Variable(torch.zeros(1, model.batch_size, model.hidden_dim)))

# Feature visualization -> Input(weights, words)
def feat_vis(h_u, words):
    labels = weights_to_2d(np.array(list(words)))
    pixmap = weights_to_2d(h_u).astype(float)
    plt.figure()
    sns.heatmap(pixmap, annot=labels, fmt = '', cmap="coolwarm", xticklabels =False, yticklabels=False)
    plt.show(block=False)

# Convert words and weights to square array for feature visualization
def weights_to_2d(weights):
    dim1 = int(np.ceil(np.sqrt(len(weights))))
    zero_pad = dim1*dim1 - len(weights) #Add zeros at end of vector if necesary to make len squared
    weights = np.pad(weights, (0,zero_pad), 'constant')
    return np.reshape(weights, (dim1, dim1))


def custom_softmax(output, T):
    return torch.exp(torch.div(output, T)) / torch.sum(torch.exp(torch.div(output, T)))
