import os
import codecs
import re
import pickle
import numpy as np

for cluster_id in range(13):
    data_dir = 'data'  # basedir
    lengthes = []  # length of paths
    all_state = []  # all paths in a sequence
    all_los = []  # los for each state in all_state
    with codecs.open(os.path.join(data_dir, 'Clusters_with_durations.txt'), encoding='utf-8') as f:
        current_cluster = 0
        for line in f:
            if re.match(r'\[(.*)\]', line) and (current_cluster == cluster_id):
                states = [m.group(1) for m in re.finditer(r'\'([A-Z]*)\'', line)]
                duration = [int(m.group(1)) for m in re.finditer(r'[\s,]+([0-9]+)[,\]]+', line)]
                all_state.extend(states)
                all_los.extend(duration)
                lengthes.append(len(states))
            elif re.match(r'Cluster ([0-9]+)', line):
                current_cluster = int(re.match(r'Cluster ([0-9]+)', line).group(1))
    lengthes = np.array(lengthes, dtype=int)

    # transforming to NP arrays
    all_los = np.array(all_los, dtype=int)
    all_state = np.array(all_state, dtype=str)

    # building state dictionary and indexing
    all_state_index = np.unique(all_state)
    all_state_id = np.array(list(map(lambda a: np.where(all_state_index == a)[0][0], all_state)), dtype=int)
    print('Processing cluster #{0} with {1} cases in states {2}'.format(cluster_id, len(lengthes), all_state_index))

    # preparing mask showing CP starting positions in all_state
    all_start_mask = [False]*len(all_state)
    s_idx = 0
    for l in lengthes:
        all_start_mask[s_idx] = True
        s_idx += l
    all_start_mask = np.array(all_start_mask)

    # count transitions between states
    transition_count = np.zeros((len(all_state_index), len(all_state_index)), dtype=int)
    for i in range(1, len(all_state)):
        if not all_start_mask[i]:
            i1 = np.where(all_state_index == all_state[i-1])
            i2 = np.where(all_state_index == all_state[i])
            transition_count[i1, i2] += 1
    print(transition_count)

    if len(lengthes) < 50:
        print('Too few cases. SKIP')
        continue

    # count transitions and probs
    ppp = []
    for i in range(len(all_state_index)):
        print('Counting transfers from  {0}... '.format(all_state_index[i]), end='')
        dst = all_state_index[transition_count[i] > 0]
        pp = []
        def get_los(src, dst):
            mask = (all_state[:-1] == src) & (all_state[1:] == dst) & ~all_start_mask[1:]
            return all_los[:-1][mask]
        t_los = [np.array(get_los(all_state_index[i], d)) for d in dst]

        if len(dst) > 0:
            t_range = np.linspace(1, max([max(dd) for dd in t_los]) - 1, 400)
            all_cases = sum([len(ld) for ld in t_los])
            p_x = [len(ld) / all_cases for ld in t_los]

            def count_transition_prob(t, los_data, p_x, los_data_idx, all_cases):
                p_t_x = sum(los_data[los_data_idx] > t) / len(los_data[los_data_idx])
                p_t = sum([sum(ld > t) for ld in los_data]) / all_cases
                return p_t_x * p_x[los_data_idx] / p_t

            for di in range(len(dst)):
                p = [count_transition_prob(t, t_los, p_x, di, all_cases) for t in t_range]
                pp.append(p)
        else:
            t_range = np.array([])
        pp = np.array(pp)
        ppp.append((all_state_index[i], dst, t_range, pp, t_los))
        print('DONE')

    # store with pickle
    with open(os.path.join('data', 'transfer_time_and_probs_cluster{0:0>2}.pkl'.format(cluster_id)), 'wb') as f:
        pickle.dump(ppp, f)
