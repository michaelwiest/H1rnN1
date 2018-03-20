from fasta_sampler_v2 import *
from RNN_v2 import *
import scipy
import IPython
'''
Usage is:
E = Evaluator(fs, model)
dist, pred, actual = E.gen_and_compare_year(10, '$MK', 2012, 10)
'''

class Evaluator(object):
    def __init__(self, fasta_sampler, model):
        self.fs = fasta_sampler
        self.model = model


    '''
    Returns a numpy array of the distance between sequences a and b.
    You should supply a 2d array of characters.
    Where each row is a sequence ie,['M', 'K', ...]
    '''
    def get_dist_matrix(self, seqs_a, seqs_b):
        if len(seqs_a.shape) == 1:
            seqs_a = np.array([seqs_a]).T
        if len(seqs_b.shape) == 1:
            seqs_b = np.array([seqs_b]).T
        seqs_a = np.vectorize(self.fs.vocabulary.get)(seqs_a)
        seqs_b = np.vectorize(self.fs.vocabulary.get)(seqs_b)
        return scipy.spatial.distance.cdist(seqs_a, seqs_b, metric='hamming')

    '''
    Dreams up a batch of guesses from the model. And compares them to the
    Following year's data (the year for which it would be predicting).
    '''
    def gen_and_compare_year(self, num_samples, primer, year, temp):
        dollar = False
        if primer.startswith('$'):
            dollar = True

        predict_len = self.fs.specified_len - len(primer)
        if dollar:
            predict_len += 1
        predictions = self.model.batch_dream(num_samples, primer, year, temp,
                                             self.fs, predict_len, split=True)
        if dollar:
            predictions = predictions[:, 1:]
        df = self.fs.to_dataframe()
        df = df[df['hemisphere'] == 'north']

        actuals = df[df['year'] == year + 1].sample(num_samples)['seq_list'].values
        actuals = np.array(actuals.tolist())
        return self.get_dist_matrix(predictions, actuals), predictions, actuals

    def gen_hamming_over_seq(self, num_samples, primer, year, temp, plot=False):
        predictions = self.model.batch_dream(num_samples, primer, year, temp,
                                     self.fs, predict_len, split=True)
        targets = generate_N_sample_per_year(num_samples, year)
        hamming_distance = []
        for i in xrange(len(seqs_a)):
            seq_a = seqs_a[0:i]
            seq_b = seqs_b[0:i]
            current_dist = scipy.spatial.distance.hamming(seqs_a, seqs_b)
            hamming_distance.append(current_dist)
        if plot:
            ind = np.arange(len(seqs_a))
            width = 0.35
            
        return hamming_distance
