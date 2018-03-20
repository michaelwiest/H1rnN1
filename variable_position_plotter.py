import matplotlib.pyplot as plt
from fasta_sampler_v2 import *
import numpy as np
plt.style.use('fivethirtyeight')
width = 0.5  # the width of the bars
thresh = 0.1



fs = FastaSamplerV2('data/HA_n_2010_2018.fa', 'data/HA_s_2010_2018.fa')
fs.set_validation_years([2016, 2017])

n_frames = []
years = list(fs.north.keys())
years.sort()

# Transition labels
ylabs = []
for i in range(1, len(years)):
    ylabs.append('{} to {}'.format(years[i-1], years[i]))



for y in years:
    temp = fs.get_AA_counts_dataframe(y)
    temp /= temp.sum(axis=1).values[0] # Normalize
    n_frames.append(temp)

s_frames = []
for y in fs.south.keys():
    temp = fs.get_AA_counts_dataframe(y, north=False)
    temp /= temp.sum(axis=1).values[0] # Normalize
    s_frames.append(temp)


changes = []
for i in range(1, len(n_frames)):
    fig, ax = plt.subplots(figsize=(12, 10))
    n_new = (np.abs(n_frames[i] - n_frames[i - 1])).sum(axis=1).values
    s_new = (np.abs(s_frames[i] - s_frames[i - 1])).sum(axis=1).values
    changes.append(n_new)
    ind = np.arange(len(n_new))
    indices = n_new > thresh
    sub_ind = ind[indices]
    sub_n = n_new[indices]
    rects1 = ax.bar(np.arange(len(sub_n)) - width/2, sub_n,
                    width, label='Northern\nHemisphere')
    rects2 = ax.bar(np.arange(len(sub_n)) + width/2, s_new[indices],
                    width, label='South\nHemisphere')
    ax.set_ylabel('Normalized change')
    ax.set_xlabel('Amino Acid Position')
    ax.set_title('Amino Change Frequency by Position\n' \
                 'Year {} to {}'.format(years[i-1], years[i]))
    ax.set_xticks(np.arange(len(sub_n)))
    ax.set_xticklabels(sub_ind, rotation=45)
    ax.legend()
    plt.show()

changes = np.array(changes)
d = pd.DataFrame(changes)
d = d.T
sub = d.loc[(d > thresh).any(axis=1)]

fig, ax = plt.subplots(figsize=(15,5))
ax.imshow(sub.T.values, cmap='hot', interpolation='nearest')
ax.set_xticks(list(range(sub.shape[0])))
ax.set_yticks(list(range(sub.shape[1])))
ax.set_yticklabels(ylabs, fontsize=10)
ax.set_xticklabels(list(sub.index.values), fontsize=10, rotation=45)
ax.grid(linewidth=0.2)
ax.set_xlabel('Amino Acid Position')
ax.set_ylabel('Year to year change')
ax.set_title('Frequency of Amino Acid Change\nfrom Year to Year by Position')
plt.show()
