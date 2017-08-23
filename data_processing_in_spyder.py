import os
import codecs
import re
import numpy as np
import hmmlearn.hmm as hmm
from sklearn.externals import joblib
import matplotlib.pyplot as plt

# reading and preprocessing data
data_dir = 'data'
lengthes = []
all_state = []
all_los = []
with codecs.open(os.path.join(data_dir, 'Clusters_with_durations.txt'), encoding='utf-8') as f:
    for line in f:
        if re.match(r'\[(.*)\]', line):
            states = [m.group(1) for m in re.finditer(r'\'([A-Z]*)\'', line)]
            duration = [int(m.group(1)) for m in re.finditer(r'[\s,]+([0-9]+)[,\]]+', line)]
            all_state.extend(states)
            all_los.extend(duration)
            lengthes.append(len(states))
            print(list(zip(states, duration)))
lengthes = np.array(lengthes, dtype=int)
all_los = np.array(all_los, dtype=int)
all_state = np.array(all_state, dtype=str)
all_state_index = np.unique(all_state)
all_state_id = np.array(list(map(lambda a: np.where(all_state_index == a)[0][0], all_state)), dtype=int)

# building sequence with q
los_quartiles = dict((aidx, np.percentile(all_los[all_state == aidx], [0, 25, 50, 75])) for aidx in all_state_index)
all_q_state = []
all_q_length = []
all_q_case_start_mask = []
t_los = 0
t_los_idx = 0
t_los_case_idx = 0
for i in range(len(all_state)):
    q_state_names = np.array(['{0}q{1}'.format(all_state[i], qi) for qi in range(4)])
    t_mask = all_los[i] > los_quartiles[all_state[i]]
    all_q_state.extend(q_state_names[t_mask])
    t_los += sum(t_mask)
    if t_los_case_idx + 1 == lengthes[t_los_idx]:
        all_q_case_start_mask.extend([True] + [False] * (t_los - 1))
        all_q_length.append(t_los)
        t_los = 0
        t_los_idx += 1
        t_los_case_idx = 0        
    else:
        t_los_case_idx += 1        
all_q_state = np.array(all_q_state)
all_q_state_index = np.unique(all_q_state)
transition_count = np.zeros((len(all_q_state_index), len(all_q_state_index)))
for i in range(1, len(all_q_state)):
    if not all_q_case_start_mask[i]:
        i1 = np.where(all_q_state_index == all_q_state[i-1])
        i2 = np.where(all_q_state_index == all_q_state[i])
        transition_count[i1, i2] +=1

# saving normed edges
tcn = transition_count / np.sum(transition_count, axis=1)[:, None]
with open('edges.csv', 'w') as f:
    f.write('Source;Target;Weight\n')
    for i in range(len(all_q_state_index)):
        for j in range(len(all_q_state_index)):
            if tcn[i,j] > 0:
                f.write('{0};{1};{2}\n'.format(all_q_state_index[i], all_q_state_index[j], tcn[i, j]))
    f.close()

# training HMM
N = 5
all_q_state_id = np.array(list(map(lambda a: np.where(all_q_state_index == a)[0][0], all_q_state)), dtype=int)
m = hmm.MultinomialHMM(n_components=N)
m.fit(all_q_state_id.reshape(-1, 1), lengths=all_q_length)

# plotting results
plt.xticks(np.arange(len(all_q_state_index)), all_q_state_index, rotation='vertical')
plt.yticks(np.arange(N))
plt.imshow(m.emissionprob_, cmap=plt.cm.hot)

