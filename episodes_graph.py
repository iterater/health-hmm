import networkx as nx
import numpy as np
import codecs
import os
import re
import difflib
import pandas as pd
import seaborn as sns
sns.set_style("whitegrid")
import matplotlib.pyplot as plt

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

# storing nodes
store_dir = '/media/iterater/DATA/Data/CPs'
with open(os.path.join(store_dir, 'nodes.csv'), 'w') as f:
    f.write('Id;Label;Cluster\n')
    f.writelines(['{0};{1};{2}\n'.format(i, all_paths[i], all_clusters[i]) for i in range(len(all_paths))])
# storing edges
boundary_similarity = 0.857
with open(os.path.join(store_dir, 'edges_unique_path_q1.csv'), 'w') as f:
    f.write('Source;Target;Weight\n')
    for i in range(len(all_paths)):
        print(i)
        for j in range(i + 1, len(all_paths)):
            similarity = difflib.SequenceMatcher(a=all_paths[i], b=all_paths[j]).ratio()
            if similarity > boundary_similarity:
                f.write('{0};{1};{2}\n'.format(i, j, similarity))

# loading and preprocessing of edges
nodes = pd.read_csv(os.path.join(store_dir, 'nodes.csv'), sep=';')                
edges = pd.read_csv(os.path.join(store_dir, 'edges.csv'), sep=';')
unique_path_cluster = nodes.groupby(['Label', 'Cluster']).size().reset_index()
unique_path_cluster.to_csv(os.path.join(store_dir, 'nodes_unique_path_cluster.csv'), sep=';')
unique_path_gr = nodes.groupby('Label')['Cluster']
unique_path_df = pd.DataFrame(index=unique_path_gr.indices)
unique_path_df['MultiCluster'] = (unique_path_gr.max() - unique_path_gr.min()) > 0
ds = unique_path_gr.min()
ds[unique_path_df['MultiCluster']] = ds.max() + 1
unique_path_df['Cluster'] = ds
unique_path_df['Size'] = unique_path_gr.count()
unique_path_df.to_csv(os.path.join(store_dir, 'nodes_unique_path.csv'), sep=';')
sns.distplot(edges['Weight'])
np.percentile(edges['Weight'], [25, 50, 75]) 
# All: Q3=0.857
# Unique CP+Cluster: Q3=0.714
# Unique CP: Q3=0.857

# graph processing
nodes = pd.read_csv(os.path.join(store_dir, 'nodes_unique_path.csv'), sep=';')                
edges = pd.read_csv(os.path.join(store_dir, 'edges_unique_path_q1.csv'), sep=';')
n_clusters = nodes['Cluster'].max() + 1
clrs = plt.get_cmap('tab20')(np.linspace(0, 1, n_clusters))
G = nx.Graph()
for idx,rec in nodes.iterrows():
    G.add_node(idx, {'color': clrs[rec['Cluster']]})
for idx,rec in edges.iterrows():
    G.add_edge(rec['Source'], rec['Target'], {'weight': rec['Weight']})
pos = nx.spring_layout(G, iterations=500, weight='weight', scale=50)
plt.figure(figsize=(10,10))S
nx.draw_networkx_edges(G, pos)
nx.draw_networkx_nodes(G, pos, list(nodes.index), 
                       node_color=list(map(lambda cl_id: clrs[cl_id], nodes['Cluster'])),
                       node_size=list(nodes['Size'] * 15))
sizelim = 40
lb_ds = nodes['Label']
lb_ds[nodes['Size'] < sizelim] = ''
nx.draw_networkx_labels(G, pos, labels=lb_ds)

# TODO: Complexity of decision tree for clusters dividing with interpretation - quality measure.
# TODO: CP structure and clinical cases relationshop