import numpy as np
import random
from torch.autograd import Variable
import torch
import pdb

# function maps each word to an index
def get_idx(char_data):
    word_to_ix = {}
    count = 0
    char_data = list(char_data)
    char_data.sort()
    for word in char_data:
        if word not in word_to_ix:
            word_to_ix[word] = count
            count += 1
    return word_to_ix

def add_cuda_to_variable(data_nums, is_gpu):
    tensor = torch.FloatTensor(data_nums)
    if isinstance(data_nums, list):
        tensor = tensor.unsqueeze_(0)
    tensor = tensor.unsqueeze_(2)
    if is_gpu:
        return Variable(tensor.cuda())[:, :, 0]
    else:
        return Variable(tensor)[:, :, 0]

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

def flip_coin_batch(probabilities, is_gpu):
    stacked_probs = np.cumsum(probabilities, axis = -1)
    N = probabilities.size(0)
    v_size = probabilities.size(1)
    rand_int = np.array([np.random.random(N)]).T
    rand_int = rand_int.repeat(v_size, axis=1)
    if is_gpu:
        sp = stacked_probs[0].cpu().numpy()
    else:
        sp = stacked_probs.numpy()
    dist = abs(sp - rand_int)
    return np.array([np.argmin(dist, axis=1)]).T

def custom_softmax(output, T):
    return torch.exp(torch.div(output, T)) / torch.sum(torch.exp(torch.div(output, T)))
