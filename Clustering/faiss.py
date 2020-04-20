import os

from utils import mkdir_if_no_exists, Timer, knn_hnsw, knn_faiss, knn_faiss_gpu, dump_data, load_data, np


# https://towardsdatascience.com/understanding-faiss-619bb6db2d1a
# Using faiss to get most similar vectors within each vectors in the instances


def build_faiss_knns(features, knn_dir, knn_method, k_neighbour, num_process=None, is_rebuild=False):

    print("Buidling faiss knn...")
    knn_prefix = os.path.join(knn_dir, '{}_k_{}'.format(knn_method, k_neighbour))
    mkdir_if_no_exists(knn_prefix)
    knn_path = knn_prefix + '.npz'
    if not os.path.isfile(knn_path) or is_rebuild:
        index_path = knn_prefix + '.index'
        with Timer('build index'):
            if knn_method == 'hnsw':
                index = knn_hnsw(features, k_neighbour, index_path)
            elif knn_method == 'faiss':
                index = knn_faiss(features, k_neighbour, index_path, omp_num_threads=num_process)
            elif knn_method == 'faiss_gpu':
                index = knn_faiss_gpu(features, k_neighbour, index_path, num_process=num_process)
            else:
                raise KeyError('Unsupported method({}). \
                        Only support hnsw and faiss currently'.format(
                    knn_method))
            knns = index.get_knns()
        with Timer('dump knns to {}'.format(knn_path)):
            dump_data(knn_path, knns, force=True)
    else:
        print('read knn from {}'.format(knn_path))
        knns = load_data(knn_path)
    return knns
