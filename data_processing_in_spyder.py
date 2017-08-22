import os
import codecs
import re
import seaborn as sns
import numpy as np
import hmmlearn.hmm as hmm
from sklearn.externals import joblib

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
all_los = np.array(all_los, dtype=int) // 15 # 15-min intervals
all_state = np.array(all_state, dtype=str)
all_state_index = np.unique(all_state)
all_state = np.array(list(map(lambda a: np.where(all_state_index == a)[0][0], 
                              all_state)), dtype=int)

x = np.transpose([all_state, all_los])

m = hmm.GaussianHMM(n_components=3)
m.fit(x, lengths=lengthes)

joblib.dump(m, 'test_model.pkl')
# TODO: discrete-events -> discrete-time