import os

from tqdm import tqdm

from Clustering.graph import Data, connected_components_constraint
from utils import Timer, clusters2labels, write_meta, read_meta, labels2clusters, np, dump_data


def filter_clusters(clusters, min_size=None, max_size=None):
    if min_size is not None:
        clusters = [c for c in clusters if len(c) >= min_size]
    if max_size is not None:
        clusters = [c for c in clusters if len(c) <= max_size]
    return clusters


def filter_knns(knns, k, th):
    pairs = []
    scores = []
    n = len(knns)
    nbrs = np.zeros([n, k], dtype=np.int32) - 1
    simi = np.zeros([n, k]) - 1
    for i, (nbr, dist) in enumerate(knns):
        assert len(nbr) == len(dist)
        nbrs[i, :len(nbr)] = nbr
        simi[i, :len(nbr)] = 1. - dist
    anchor = np.tile(np.arange(n).reshape(n, 1), (1, k))

    # filter
    selidx = np.where((simi >= th) & (nbrs != -1) & (nbrs != anchor))
    pairs = np.hstack((anchor[selidx].reshape(-1,
                                              1), nbrs[selidx].reshape(-1, 1)))
    scores = simi[selidx]

    # keep uniq pairs
    pairs = np.sort(pairs, axis=1)
    pairs, unique_idx = np.unique(pairs, return_index=True, axis=0)
    scores = scores[unique_idx]
    return pairs, scores


def graph_clustering_dynamic_th(edges, score, max_sz, th_step, max_iter):
    edges = np.sort(edges, axis=1)
    th = score.min()

    # construct graph
    score_dict = {}  # score lookup table
    for i, e in enumerate(edges):
        score_dict[e[0], e[1]] = score[i]
    nodes = np.sort(np.unique(edges.flatten()))
    mapping = -1 * np.ones((nodes.max() + 1), dtype=np.int)
    mapping[nodes] = np.arange(nodes.shape[0])
    link_idx = mapping[edges]

    # print('nodes: ', nodes, 'vertex shape: ', nodes.shape)

    vertex = [Data(n) for n in nodes]
    # print('vertex: ', vertex[0].__dict__, 'vertex size: ', vertex.__sizeof__())

    for l, s in zip(link_idx, score):
        vertex[l[0]].add_link(vertex[l[1]], s)
    # print('vertex[l: ', list(vertex[0].__dict__.get('_Data__links'))[0].__dict__, 'vertex size: ', vertex.__sizeof__())

    # first iteration
    comps, remain = connected_components_constraint(vertex, max_sz)
    # print('comps: ', list(comps[0])[0].__dict__)
    # print('remain: ', list(remain)[0].__dict__)
    # iteration
    components = comps[:]
    Iter = 0
    while remain:
        th = th + (1 - th) * th_step
        comps, remain = connected_components_constraint(remain, max_sz, score_dict, th)
        components.extend(comps)
        # print('comps: ', list(comps[0])[0].__dict__)
        Iter += 1
        if Iter >= max_iter:
            # print("\t Force stopping at: th {}, remain {}".format(th, len(remain)))
            components.append(remain)
            remain = {}
    # print('components: ', list(components[0])[0].__dict__)
    return components


def generate_cluster(proposal_dir, porposal_label_path, faiss_knns, k_neighbour, th_knn, th_step, max_size, min_size,
                     max_iter):
    print("Generating cluster...")
    if not os.path.exists(proposal_dir):
        os.makedirs(proposal_dir)
    if not os.path.isfile(porposal_label_path):
        with Timer('build super vertices'):

            pairs, scores = filter_knns(faiss_knns, k_neighbour, th_knn) # Prune edge, filter out edges lower than th_knn
            comps = graph_clustering_dynamic_th(pairs, scores, max_size, th_step, max_iter=max_iter) #super vertex generation
            clusters = [sorted([n.name for n in c]) for c in comps]

        with Timer('dump clustering to {}'.format(porposal_label_path)):
            labels = clusters2labels(clusters)
            write_meta(porposal_label_path, labels)
    else:
        lb2idxs, _ = read_meta(porposal_label_path)
        clusters = labels2clusters(lb2idxs)
    clusters = filter_clusters(clusters, min_size)
    return clusters


def save_proposals(clusters, knns, proposal_dir, force=True):
    print('saving cluster proposals to {}'.format(proposal_dir))
    for lb, nodes in enumerate(tqdm(clusters)):
        nodes = set(nodes)
        edges = []
        visited = set()
        # get edges from knn
        for idx in nodes:
            ners, dists = knns[idx]
            for n, dist in zip(ners, dists):
                if n == idx or n not in nodes:
                    continue
                idx1, idx2 = (idx, n) if idx < n else (n, idx)
                key = '{}-{}'.format(idx1, idx2)
                if key not in visited:
                    visited.add(key)
                    edges.append([idx1, idx2, dist])
        # save to npz file
        opath_node = os.path.join(proposal_dir, '{}_node.npz'.format(lb))
        opath_edge = os.path.join(proposal_dir, '{}_edge.npz'.format(lb))
        nodes = list(nodes)
        dump_data(opath_node, data=nodes, force=force)
        dump_data(opath_edge, data=edges, force=force)


