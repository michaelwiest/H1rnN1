from pdb import set_trace as bp
import numpy as np
from Bio import SeqIO
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import scipy
from helper import get_idx
from collections import Counter
from scipy.spatial.distance import *

'''
Class for handling fasta files. It essentially generates random combinations
of AA sequences from the specified years. Currently can only generate
AA sequences from a winter > summer > winter combination.
'''
class FastaSampler(object):
    def __init__(self, north_fasta, south_fasta,
                 start='$', end='%', delim0='&', delim1='@', pad_char='_',
                 specified_len=566):
        self.start = start
        self.end = end
        self.delim0 = delim0
        self.delim1 = delim1
        self.pad_char = pad_char
        self.df = None
        self.train_years = None
        self.validation_years = None
        self.specified_len = specified_len
        self.handle_files(north_fasta, south_fasta)
        self.char_lookup = np.vectorize(self.vocabulary.get)

    def handle_files(self, north_fasta, south_fasta):
        self.north, v1 = self.__parse_fasta_to_list(north_fasta, 'north')
        self.south, v2 = self.__parse_fasta_to_list(south_fasta, 'south')
        vocab_temp = ''.join(list(set(list(v1) + list(v2))))
        self.all_aas = vocab_temp
        self.__generate_vocabulary(vocab_temp)
        self.df = self.instantiate_dataframe()

    def __generate_vocabulary(self, vocabulary):
        t = len(vocabulary)
        vocabulary += self.start
        vocabulary += self.end
        # vocabulary += self.delim0
        # vocabulary += self.delim1
        self.vocabulary = get_idx(vocabulary)
        self.num_special_chars = len(self.vocabulary) - t

        # This is for the zero padding character.
        # self.vocabulary[self.pad_char] = 0
        self.inverse_vocabulary = {v: k for k, v in self.vocabulary.items()}

    def __parse_fasta_to_list(self, some_fasta, area):
        fasta_sequences = SeqIO.parse(open(some_fasta),'fasta')
        data = {}
        num_missing = 0
        num_too_long = 0
        seqs = set()
        # Basic data structure for our samples.

        for f in fasta_sequences:
            template = {
                        'id': '',
                        'year': '',
                        'month': '',
                        'day': '',
                        'location': '',
                        'seq': ''
                         }
            desc = f.description
            desc_split = desc.split(' ')

            date = desc_split[-2]
            date_split = date.split('/')
            year = int(date_split[0])
            try:
                month = int(date_split[1])
                day = int(date_split[2])
            except ValueError:
                num_missing += 1
                continue

            if len(f.seq) != self.specified_len:
                num_too_long += 1
                continue

            location = desc_split[1].split('/')[1]
            seq = str(f.seq)
            [seqs.add(s) for s in list(seq)]

            template['id'] = f.id
            template['year'] = year
            template['month'] = month
            template['day'] = day
            template['location'] = location
            template['seq'] = seq
            template['seq_list'] = list(seq)
            template['hemisphere'] = area


            if year not in data.keys():
                data[year] = []
            data[year].append(template)

        print('Missing data: {}'.format(num_missing))
        print('Bad length data: {}'.format(num_too_long))
        # self.__generate_vocabulary(''.join(list(seqs)))

        return data, ''.join(list(seqs))

    def set_validation_years(self, validation):
        all_years = list(set(self.df.year.values))
        self.train_years = list(set(all_years) - set(validation))
        self.validation_years = list(set(all_years) - set(self.train_years))

        # Get rid of first year because it actually can't be sampled From
        # Because there is no earlier year.
        self.train_years.sort()
        self.train_years = self.train_years[1:]
        self.validation_years.sort()
        self.validation_years = self.validation_years[:-1]

        # Update the rows in the dataframe.
        self.df['train'] = False
        self.df.loc[self.df.year.isin(self.train_years), 'train'] = True

    def generate_N_random_samples_and_targets(self, N, group='train',
                                              slice_len=None,
                                              to_num=True,
                                              shift_index=True):
        if self.train_years is None:
            raise ValueError('Please set train and validation years first')
        num_samples_so_far = 0
        first = True
        while num_samples_so_far < N:
            num_samples = np.random.randint(1, N - num_samples_so_far + 1)
            if group.lower() == 'train':
                year = self.train_years[np.random.randint(len(self.train_years))]
            elif group.lower() == 'validation':
                year = self.validation_years[np.random.randint(len(self.validation_years))]
            new_vals = self.generate_N_sample_per_year(num_samples, year,
                                                       to_num=to_num)
            if first:
                output = new_vals
            else:
                output = np.concatenate((output, new_vals), axis=1)
            num_samples_so_far += num_samples
            first = False

        output = output.transpose(1, 2, 0)
        print(output.shape)
        min2 = output[:, 0, :]
        min1 = output[:, 1, :]
        min0 = output[:, 2, :]
        target = output[:, 2, :]

        if slice_len is not None:
            if to_num:
                min0_slice = np.zeros((min0.shape[0], slice_len))
                targets_slice = np.zeros((min0.shape[0], slice_len))
            else:
                min0_slice = np.empty((min0.shape[0], slice_len), dtype=str)
                targets_slice = np.empty((min0.shape[0], slice_len), dtype=str)
            indices = np.random.randint(max(1, min0.shape[1] - slice_len), size=N)
            for i, index in enumerate(indices):
                if shift_index:
                    min0_slice[i, :] = min0[i, index: index + slice_len]
                    targets_slice[i, :] = min0[i, index + 1: index + slice_len + 1]
                else:
                    min0_slice[i, :] = min0[i, index: index + slice_len]
                    targets_slice[i, :] = min0_slice[i, :]

            target = targets_slice
            min0 = min0_slice
        return [min2, min1], min0, target


    def generate_N_sample_per_year(self,
                                   N,
                                   year,
                                   full=True,
                                   to_num=True,
                                   pattern=['w', 'w', 'w']
                                   ):
        '''
        If you want samples from the 2012/2013 winter, 2013 summer, and 2014 winter,
        supply 2013 as the year. This returns an Nx3 array. Where the pattern
        supplied is represented in each row.
        '''
        if year not in self.north.keys() or year not in self.south.keys() or \
                year + 1 not in self.north.keys() or year - 1 not in self.north.keys():
            raise ValueError('Specified year ({}) is not present in dataset.\n' \
                             'Maximum year is: {}'.format(year, max(self.north.keys())))
        if len(pattern) > 3:
            raise ValueError('Please only supply patterns of length 3')
        # Months bounding the flu season. w = winter. and s = summer for
        # southern hemisphere.
        w_upper = 5
        w_lower = 10
        s_upper = 10
        s_lower = 5

        to_return = []
        all_seqs = []
        current_year = year
        # This are for selecting appropriately close distances.
        previous_seqs = None
        for i, p in enumerate(pattern):
            if not i == 0 and not (p.lower() == 's' and pattern[i-1].lower() == 'w'):
                current_year += 1
            if p.lower() == 'w':
                winters = self.df.loc[self.df.hemisphere == 'north']

                possible_winters = winters.loc[((winters.month <= w_upper) &
                                               (winters.year == current_year)) |
                                               ((winters.month >= w_lower) &
                                               (winters.year == current_year - 1))]

                previous_seqs = self.__get_sequences_within_dist_of_seq(N,
                                                                        previous_seqs,
                                                                        possible_winters)

            elif p.lower() == 's':
                summers = self.df.loc[self.df.hemisphere == 'south']

                possible_summers = summers.loc[(summers.month <= s_upper) &
                                               (summers.year == current_year) &
                                               (summers.month >= s_lower)]
                previous_seqs = self.__get_sequences_within_dist_of_seq(N,
                                                                        previous_seqs,
                                                                        possible_summers)
            all_seqs.append(previous_seqs)

        all_seqs = np.array(all_seqs).T

        if to_num:
            to_return = self.char_lookup(all_seqs)
        return to_return

    def __get_sequences_within_dist_of_seq(self, N, arg_seqs, arg_df,
                                           distance=0.002):

        if arg_seqs is not None:
            df_vals = arg_df[list(range(0, self.specified_len))].values
            df_ints = self.char_lookup(df_vals)
            arg_seqs = self.char_lookup(arg_seqs)

            dists = scipy.spatial.distance.cdist(arg_seqs, df_ints,
                                                 metric='hamming')
            bool_dists = (dists < distance).astype(int)
            samples = []

            for sample in range(dists.shape[0]):
                row_probs = bool_dists[sample, :] / bool_dists[sample, :].sum()
                which_sample = np.random.choice(dists.shape[1], size=1,
                                                p=row_probs)
                samples.append(df_vals[which_sample, :])
            samples = np.array(samples).squeeze(1)
        else:
            sub = arg_df.sample(N)
            samples = sub[list(range(0, self.specified_len))].values
        return samples



    def get_AA_counts_by_position(self, year, north=True, plot=False):
        '''
        Returns a dictionary where the keys are amino acids and the
        values are a list of the count of that amino acid at the given
        position across all samples. Way less useful than the funciton below.
        '''
        if north:
            data = self.north[year]
        else:
            data = self.south[year]

        AAs = set(''.join([d['seq'] for d in data]))
        to_return = {}
        for aa in AAs:
            to_return[aa] = [0] * len(data[0]['seq'])

        for i in range(len(data)):
            sample = data[i]
            for j in range(len(sample['seq'])):
                aa = sample['seq'][j]
                to_return[aa][j] += 1

        if plot:
            ind = np.arange(len(data[0]['seq']))
            width = 0.35
            prev = [0] * len(ind)
            for aa, counts in to_return.items():
                plt.bar(ind, counts, width, label=aa, bottom=prev)
                prev += counts
            plt.show()

        return to_return


    def get_AA_counts_dataframe(self, year, north=True, plot=False):
        '''
        Get pandas dataframe with counts of each AA by position.
        '''
        if north:
            data = self.north[year]
        else:
            data = self.south[year]

        alphabet = ''.join(list(set(''.join([s['seq'] for s in data]))))
        all_seq = [s['seq'] for s in data]
        char_to_int = dict((c, i) for i, c in enumerate(alphabet))
        integer_encoded = [[char_to_int[char] for char in d] for d in all_seq]

        # AAs = list(set(''.join([d['seq'] for d in data])))
        AAs = self.all_aas
        to_return = np.zeros((len(data[0]['seq']), len(AAs)))

        for i in range(len(data)):
            sample = data[i]['seq']
            values = list(sample)
            if i == 0:
                label_encoder = LabelEncoder()
                label_encoder.fit(list(AAs))
                labs = label_encoder.classes_
                onehot_encoder = OneHotEncoder(sparse=False, n_values=len(AAs))
            integer_encoded = label_encoder.transform(values)
            integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
            if i == 0:
                onehot_encoded = onehot_encoder.fit(integer_encoded)
            onehot_encoded = onehot_encoder.transform(integer_encoded)
            to_return += onehot_encoded
        to_return = pd.DataFrame(to_return, index=np.arange(len(data[0]['seq'])), columns=labs)
        return to_return

    def get_freq_sequence(self, year, north=True, plot=False):
        if north:
            data = self.north[year]
        else:
            data = self.south[year]
        alphabet = ''.join(list(set(''.join([s['seq'] for s in data]))))
        all_seq = [s['seq'] for s in data]
        data = Counter(all_seq)
        seq_freq = data.most_common()
        max_freq = data.most_common(1)
        return seq_freq, max_freq

    def get_score(self, seq0, seq1):
        return hamming(seq0, seq1)


    def instantiate_dataframe(self, just_vals=False):

        self.df = pd.DataFrame(columns=['id', 'hemisphere',
                                        'year', 'month',
                                        'day', 'location',
                                        'seq', 'seq_list'
                                        ])
        for year, vals in self.north.items():
            self.df = self.df.append(pd.DataFrame(self.north[year]))
        for year, vals in self.south.items():
            self.df = self.df.append(pd.DataFrame(self.south[year]))
        to_add = pd.DataFrame(self.df.seq_list.values.tolist(),
                              index=self.df.index)

        # for col in to_add.columns:
        #     to_add[col] = to_add[col].astype('category')

        self.df[list(range(self.specified_len))] = to_add

        self.df.index = self.df.id

        self.df['train'] = False

        if just_vals:
            return self.df[list(range(self.specified_len))]
        else:
            return self.df


    def get_flu_sequences_for_year(self, year, north=True):
        w_upper = 5
        w_lower = 10
        s_upper = 10
        s_lower = 5
        df = self.df
        if north:
            df = df[df['hemisphere'] == 'north']
            sub = df[((df['year'] == year) & (df['month'] <= w_upper)) |
                     ((df['year'] == year - 1) & (df['month'] >= w_lower))]
        else:
            df = df[df['hemisphere'] == 'south']
            sub = df[((df['year'] == year) & (df['month'] <= s_upper)) &
                     ((df['year'] == year) & (df['month'] >= s_lower))]
        return sub[list(range(self.specified_len))]

    def get_count_matrix_from_sequences(self, sequence_array):
        # to_return = pd.DataFrame(columns=list(self.all_aas),
        #                          index=list(range(self.specified_len)))
        df = pd.DataFrame(sequence_array)
        out = [dict(df[i].value_counts()) for i in list(df.columns.values)]
        return pd.DataFrame.from_dict(out).fillna(0)





pass
