from fasta_sampler_v2 import *
from RNN_v2 import *
import scipy
<<<<<<< HEAD
<<<<<<< HEAD
'''
Usage is:
E = Evaluator(fs, model)
dist, pred, actual = E.gen_and_compare_year(10, '$MK', 2012, 10)
'''
=======

>>>>>>> 922682fee0770ca7ca29849da1798f09b06f40e1
=======

>>>>>>> master

class Evaluator(object):
    def __init__(self, fasta_sampler, model):
        self.fs = fasta_sampler
        self.model = model


<<<<<<< HEAD
<<<<<<< HEAD
    '''
    Returns a numpy array of the distance between sequences a and b.
    You should supply a 2d array of characters.
    Where each row is a sequence ie,['M', 'K', ...]
    '''
=======
>>>>>>> 922682fee0770ca7ca29849da1798f09b06f40e1
=======
>>>>>>> master
    def get_dist_matrix(self, seqs_a, seqs_b):
        if len(seqs_a.shape) == 1:
            seqs_a = np.array([seqs_a]).T
        if len(seqs_b.shape) == 1:
            seqs_b = np.array([seqs_b]).T
        seqs_a = np.vectorize(self.fs.vocabulary.get)(seqs_a)
        seqs_b = np.vectorize(self.fs.vocabulary.get)(seqs_b)
        return scipy.spatial.distance.cdist(seqs_a, seqs_b, metric='hamming')

<<<<<<< HEAD
<<<<<<< HEAD
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
            predictions = predictions[:,1:]
        df = self.fs.to_dataframe()

        actuals = df[df['year'] == year + 1].sample(num_samples)['seq_list'].values
        actuals = np.array(actuals.tolist())
        return self.get_dist_matrix(predictions, actuals), predictions, actuals
=======
=======
>>>>>>> master
    def gen_and_compare_year(self, num_samples, primer, year, temp,
                             predict_len):
        predictions = self.model.batch_dream(num_samples, primer, year, temp,
                                             self.fs, predict_len)
        df = self.fs.get_dataframe()
        actuals = df[df['year'] == year + 1].sample(num_samples)['seq_list'].values
        actuals = np.array(actuals.tolist())
        return self.get_dist_matrix(predictions, actuals)
<<<<<<< HEAD
>>>>>>> 922682fee0770ca7ca29849da1798f09b06f40e1
=======
>>>>>>> master
