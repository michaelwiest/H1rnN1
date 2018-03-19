import numpy as np
import random
from torch.autograd import Variable
import torch
import pdb

# function maps each word to an index
def get_idx(char_data):
    word_to_ix = {}
    count = 0
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

def custom_softmax(output, T):
    return torch.exp(torch.div(output, T)) / torch.sum(torch.exp(torch.div(output, T)))

def custom_softmax2(output, T):
    exp_i = torch.exp(output/ T)
    divider = torch.sum(exp_i,1)
    divider = divider.expand(output.size(1), len(divider)).transpose(0,1)
    return exp_i/ divider
