import os
import codecs
import re

data_dir = 'D:\Src\health-hmm\data'
with codecs.open(os.path.join(data_dir, 'Clusters_with_durations.txt'), encoding='utf-8') as f:
    for line in f:
        if re.match(r'\[(.*)\]', line):
            states = [m.group(1) for m in re.finditer(r'\'([A-Z]*)\'', line)]
            duration = [int(m.group(1)) for m in re.finditer(r'[\s,]+([0-9]+)[,\]]+', line)]
            print(states, duration)
            # s = re.search(r'\[(.*)\]', line).group(1)
            # observations = np.array(list(map(float, re.split(r'\s*,\s*', s))), dtype=float)
