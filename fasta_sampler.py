import numpy as np
from Bio import SeqIO

'''
Class for handling fasta files. It essentially generates random combinations
of AA sequences from the specified years. Currently can only generate
AA sequences from a winter > summer > winter combination.
'''
class FastaSampler(object):
    def __init__(self, north_fasta, south_fasta,
                 start='$', end='%', delim0='&', delim1='@'):
        self.handle_files(north_fasta, south_fasta)
        self.start = start
        self.end = end
        self.delim0 = delim0
        self.delim1 = delim1

    def handle_files(self, north_fasta, south_fasta):
        self.north = self.__parse_fasta_to_list(north_fasta)
        self.south = self.__parse_fasta_to_list(south_fasta)

    def __parse_fasta_to_list(self, some_fasta, delim='>'):
        fasta_sequences = SeqIO.parse(open(some_fasta),'fasta')
        data = {}
        num_missing = 0
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

            location = desc_split[1].split('/')[1]

            template['id'] = f.id
            template['year'] = year
            template['month'] = month
            template['day'] = day
            template['location'] = location
            template['seq'] = str(f.seq)

            if year not in data.keys():
                data[year] = []
            data[year].append(template)
        print('Missing data: {}'.format(num_missing))
        return data


    # If you want samples from the 2012/2013 winter, 2013 summer, and 2014 winter,
    # supply 2013 as the year.
    def generate_N_sample(self, N, year, full=True):
        if year not in self.north.keys() or year not in self.south.keys() or \
                year + 1 not in self.north.keys() or year - 1 not in self.north.keys():
            raise ValueError('Specified year ({}) is not present in dataset.\n' \
                             'Maximum year is: {}'.format(year, max(self.north.keys()) - 1))
        w_upper = 5
        w_lower = 10

        s_upper = 10
        s_lower = 5
        to_return = []
        prev_winter_seq = []
        prev_summer_seq = []
        future_winter_seq = []

        possible_prev_winters = self.north[year] + self.north[year - 1]
        possible_prev_summers = self.south[year]
        possible_future_winters = self.north[year] + self.north[year + 1]

        # Get N of the previous winters possible sequences.
        while len(prev_winter_seq) < N:
            ind = np.random.randint(len(possible_prev_winters))
            sample = possible_prev_winters[ind]
            if (sample['year'] == year and sample['month'] <= w_upper) or \
                    (sample['year'] == year - 1 and sample['month'] >= w_lower):
                prev_winter_seq.append(sample['seq'])

        # Get N summer sequences from southern.
        while len(prev_summer_seq) < N:
            ind = np.random.randint(len(possible_prev_summers))
            sample = possible_prev_summers[ind]
            if (sample['year'] == year and sample['month'] <= s_upper and \
                    sample['month'] >= s_lower):
                prev_summer_seq.append(sample['seq'])

        # Get N future winter sequences.
        while len(future_winter_seq) < N:
            ind = np.random.randint(len(possible_future_winters))
            sample = possible_future_winters[ind]
            if (sample['year'] == year + 1 and sample['month'] <= w_upper) or \
                    (sample['year'] == year and sample['month'] >= w_lower):
                future_winter_seq.append(sample['seq'])

        # If we want the full sequence (for training) vs. if we only want
        # the first two portions (for priming for generation).
        if full:
            return [self.start + prev_winter_seq[i] + self.delim0 +
                    prev_summer_seq[i] + self.delim1 + future_winter_seq[i] +
                    self.end for i in range(N)]
        else:
            return [self.start + prev_winter_seq[i] + self.delim0 +
                    prev_summer_seq[i] + self.delim1 for i in range(N)]


    def get_AA_variability(self, year, north=True):
        if north:
            data = self.north[year]
        else:
            data = self.south[year]
        to_return = []
        # to_return = [{}] * len(data[0]['seq'])
        to_return = []
        for i in range(len(data)):
            sample = data[i]
            for j in range(len(sample['seq'])):
                if i == 0:
                    to_return.append({})
                if sample['seq'][j] not in to_return[j].keys():
                    to_return[j][sample['seq'][j]] = 1
                else:
                    to_return[j][sample['seq'][j]] += 1
            # break
        return to_return






pass
