import networkx as nx
import numpy as np
import codecs
import os
import re
import difflib

data_dir = 'data'  # basedir
all_paths = []
all_clusters = []
with codecs.open(os.path.join(data_dir, 'Clusters_with_durations.txt'), encoding='utf-8') as f:
    current_cluster = 0
    for line in f:
        if re.match(r'\[(.*)\]', line):
            states = [m.group(1) for m in re.finditer(r'\'([A-Z]*)\'', line)]
            all_paths.append(''.join(states))
            all_clusters.append(current_cluster)
        elif re.match(r'Cluster ([0-9]+)', line):
            current_cluster = int(re.match(r'Cluster ([0-9]+)', line).group(1))

# storing data
with open(os.path.join(data_dir, 'nodes.csv'), 'w') as f:
    f.write('Id;Label;Cluster\n')
    f.writelines(['{0};{1};{2}\n'.format(i, all_paths[i], all_clusters[i]) for i in range(len(all_paths))])
with open(os.path.join(data_dir, 'edges.csv'), 'w') as f:
    f.write('Source;Target;Weight\n')
    for i in range(len(all_paths)):
        print(i)
        for j in range(i + 1, len(all_paths)):
            similarity = difflib.SequenceMatcher(a=all_paths[i], b=all_paths[j]).ratio()
            f.write('{0};{1};{2}\n'.format(i, j, similarity))

# graph processing
N = 1000
G = nx.Graph()
G.add_nodes_from(np.arange(N))
for i in range(N):
    print(i)
    for j in range(i + 1, N):
        if difflib.SequenceMatcher(a=all_paths[i], b=all_paths[j]).ratio() > 0.7:            
            G.add_edge(i,j)
nx.draw(G)
