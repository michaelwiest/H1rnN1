from fasta_sampler import *
import numpy as np
from sklearn.decomposition import PCA

fs = FastaSampler('data/HA_n_2010_2018.fa', 'data/HA_s_2010_2018.fa')

years = fs.north.keys()
n_seq_year = []
s_seq_year = []
north_seq_year = []
south_seq_year = []

print fs.handle_files('data/HA_n_2010_2018.fa', 'data/HA_s_2010_2018.fa')

for i in years:
    for j in xrange(len(fs.north[i])):
        north_seq = fs.north[i][j]['seq']
        nor_seq_num = 2
        n_seq_year.append(nor_seq_num)
    north_seq_year.append(n_seq_year)

for i in years:
    for j in xrange(len(fs.south[i])):
        south_seq = fs.south[i][j]['seq']
        south_seq_num = [ord(char) - 96 for char in south_seq]
        s_seq_year.append(south_seq_num)
    south_seq_year.append(s_seq_year)

#print north_seq_year[0]
#pca = PCA(n_components=2)
#pca.fit(north_seq_year)
