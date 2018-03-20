from fasta_sampler_v2 import *
from RNN_v2 import *
from scipy.spatial.distance import *

class Evaluator(object):
    def __init__(self, fasta_sampler, model):
        self.fs = fasta_sampler
        self.model = model


    def get_dist_matrix(self, seqs_a, seqs_b):
        seqs_a = np.vectorize(self.fs.vocabulary.get)(seqs_a)
        seqs_b = np.vectorize(self.fs.vocabulary.get)(seqs_b)
        return scipy.spatial.distance.cdist(seqs_a, seqs_b, metric='hamming')

    def gen_and_compare_year(self, num_samples, primer, year, temp,
                             predict_len):
        predictions = self.model.batch_dream(num_samples, primer, year, temp,
                                             self.fs, predict_len)
        df = self.fs.get_dataframe()
        actuals = df[df['year'] == year].sample(num_samples)['seq'].values
        return self.get_dist_matrix(predictions, actuals)
