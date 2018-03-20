from fasta_sampler_v2 import *
from RNN_v2 import *
import scipy


class Evaluator(object):
    def __init__(self, fasta_sampler, model):
        self.fs = fasta_sampler
        self.model = model


    def get_dist_matrix(self, seqs_a, seqs_b):
        if len(seqs_a.shape) == 1:
            seqs_a = np.array([seqs_a]).T
        if len(seqs_b.shape) == 1:
            seqs_b = np.array([seqs_b]).T
        seqs_a = np.vectorize(self.fs.vocabulary.get)(seqs_a)
        seqs_b = np.vectorize(self.fs.vocabulary.get)(seqs_b)
        return scipy.spatial.distance.cdist(seqs_a, seqs_b, metric='hamming')

    def gen_and_compare_year(self, num_samples, primer, year, temp,
                             predict_len):
        predictions = self.model.batch_dream(num_samples, primer, year, temp,
                                             self.fs, predict_len)
        df = self.fs.get_dataframe()
        actuals = df[df['year'] == year + 1].sample(num_samples)['seq_list'].values
        actuals = np.array(actuals.tolist())
        return self.get_dist_matrix(predictions, actuals)
