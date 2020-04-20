#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import json
import os
import pickle
import random
import shutil
import time

import numpy as np
import scipy.sparse as sp
import torch


class TextColors:
    HEADER = '\033[35m'
    OKBLUE = '\033[34m'
    OKGREEN = '\033[32m'
    WARNING = '\033[33m'
    FATAL = '\033[31m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class Timer():
    def __init__(self, name='task', verbose=True):
        self.name = name
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.verbose:
            print('[Time] {} consumes {:.4f} s'.format(
                self.name,
                time.time() - self.start))
        return exc_type is None


def set_random_seed(seed, cuda=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


def is_l2norm(features, size):
    rand_i = random.choice(range(size))
    norm_ = np.dot(features[rand_i, :], features[rand_i, :])
    return abs(norm_ - 1) < 1e-6


def row_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    # if rowsum <= 0, keep its previous value
    rowsum[rowsum <= 0] = 1
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def is_spmat_eq(a, b):
    return (a != b).nnz == 0


def aggregate(features, adj, times):
    dtype = features.dtype
    for i in range(times):
        features = adj * features
    return features.astype(dtype)


def read_probs(path, inst_num, feat_dim, dtype=np.float32, verbose=False):
    assert (inst_num > 0 or inst_num == -1) and feat_dim > 0
    count = -1
    if inst_num > 0:
        count = inst_num * feat_dim  # 18000 * 20
        print(count)
    # probs = np.fromfile(path, dtype=dtype, count=count)
    # probs = np.load(path, allow_pickle=True)
    probs = np.loadtxt(path, delimiter=' ')
    # probs = np.ravel(probs)
    if feat_dim > 1:
        probs = probs.reshape(inst_num, feat_dim)
    if verbose:
        print('[{}] shape: {}'.format(path, probs.shape))
    return probs


def read_meta(fn_meta, start_pos=0, verbose=True):
    lb2idxs = {}  # instances
    idx2lb = {}  # classes, contains array of line number of the instances found matching class number
    with open(fn_meta) as f:
        for idx, x in enumerate(f.readlines()[start_pos:]):  # idx = counter, x = value
            lb = int(x.strip())
            if lb not in lb2idxs:
                lb2idxs[lb] = []
            lb2idxs[lb] += [idx]
            idx2lb[idx] = lb

    inst_num = len(idx2lb)
    cls_num = len(lb2idxs)
    if verbose:
        print('[{}] #cls: {}, #inst: {}'.format(fn_meta, cls_num, inst_num))
    return lb2idxs, idx2lb


def write_meta(ofn, idx2lb, inst_num=None):
    print('save label to', ofn)
    if inst_num is None:
        inst_num = max(idx2lb.keys()) + 1
    cls_num = len(set(idx2lb.values()))
    with open(ofn, 'w') as of:
        current_lb = 0
        discard_lb = 0
        map2newlb = {}
        for idx in range(inst_num):
            if idx in idx2lb:
                lb = idx2lb[idx]
                if lb in map2newlb:
                    new_lb = map2newlb[lb]
                else:
                    new_lb = current_lb
                    map2newlb[lb] = new_lb
                    current_lb += 1
            else:
                new_lb = cls_num + discard_lb
                discard_lb += 1
            of.write(str(new_lb) + '\n')
    assert current_lb == cls_num, '{} vs {}'.format(current_lb, cls_num)

    print('#discard: {}, #lbs: {}'.format(discard_lb, current_lb))
    print('#inst: {}, #class: {}'.format(inst_num, cls_num))


def write_feat(ofn, features):
    print('save features to', ofn)
    features.tofile(ofn)


def dump2npz(ofn, data, force=False):
    if os.path.exists(ofn) and not force:
        return
    np.savez_compressed(ofn, data=data)


def dump2json(ofn, data, force=False):
    if os.path.exists(ofn) and not force:
        return

    def default(obj):
        if isinstance(obj, np.int32):
            return int(obj)
        elif isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, set) or isinstance(obj, np.ndarray):
            return list(obj)
        else:
            raise TypeError("Unserializable object {} of type {}".format(
                obj, type(obj)))

    with open(ofn, 'w') as of:
        json.dump(data, of, default=default)


def dump2pkl(ofn, data, force=False):
    if os.path.exists(ofn) and not force:
        return
    with open(ofn, 'wb') as of:
        pickle.dump(data, of)


def dump_data(ofn, data, force=False):
    if os.path.exists(ofn) and not force:
        print('{} already exists. Set force=True to overwrite.'.format(ofn))
        return
    mkdir_if_no_exists(ofn)
    if ofn.endswith('.json'):
        dump2json(ofn, data, force=force)
    elif ofn.endswith('.pkl'):
        dump2pkl(ofn, data, force=force)
    else:
        dump2npz(ofn, data, force=force)


def load_npz(fn):
    return np.load(fn)['data.npy']


def load_pkl(fn):
    return pickle.load(open(fn, 'rb'))


def load_json(fn):
    return json.load(open(fn, 'r'))


def load_data(ofn):
    if ofn.endswith('.json'):
        return load_json(ofn)
    elif ofn.endswith('.pkl'):
        return load_pkl(ofn)
    else:
        return load_npz(ofn)


def labels2clusters(lb2idxs):
    clusters = [idxs for _, idxs in lb2idxs.items()]
    return clusters


def clusters2labels(clusters):
    idx2lb = {}
    for lb, cluster in enumerate(clusters):
        for v in cluster:
            idx2lb[v] = lb
    return idx2lb


def mkdir_if_no_exists(path, subdirs=[''], is_folder=False):
    if path == '':
        return
    for sd in subdirs:
        if sd != '' or is_folder:
            d = os.path.dirname(os.path.join(path, sd))
        else:
            d = os.path.dirname(path)
        if not os.path.exists(d):
            os.makedirs(d)


def rm_suffix(s):
    return s[:s.rfind(".")]


def rand_argmax(v):
    assert len(v.squeeze().shape) == 1
    return np.random.choice(np.flatnonzero(v == v.max()))


def create_temp_file_if_exist(path, suffix=''):
    path_with_suffix = path + suffix
    if not os.path.exists(path_with_suffix):
        return path_with_suffix
    else:
        i = 0
        while i < 1000:
            temp_path = '{}_{}'.format(path, i) + suffix
            i += 1
            if not os.path.exists(temp_path):
                return temp_path


def line2label(label_file_path, line2label_file_path):
    dict = []
    with open(label_file_path, 'r') as file:
        read = file.readlines()
        for no, line in enumerate(read):
            dict.append([no, int(line.replace('\n', ''))])
        file.close()

    with open(line2label_file_path, 'w') as l2lfile:
        writel2l = csv.writer(l2lfile)
        for item in dict:
            writel2l.writerow(item)
        l2lfile.close()


def reorder(line2label_file_path, feature_file_path):
    featurearr = []
    orderedpairarr = []
    with open(line2label_file_path, 'r') as file:
        read = csv.reader(file)
        line2label_sortedlist = sorted(read, key=lambda row: int(row[1]))
        file.close()

    with open(feature_file_path, 'r') as file:
        read = csv.reader(file)
        for row in read:
            featurearr.append(row)
        file.close()

    for items in line2label_sortedlist:
        orderedpairarr.append([featurearr[int(items[0])], items[1]])

    with open('Input/input/orderedfeature.csv', 'w') as file:
        write = csv.writer(file)
        for items in orderedpairarr:
            write.writerow(items[0])
        file.close()

    with open('Input/input/orderedlabel.meta', 'w') as file:
        write = csv.writer(file)
        for items in orderedpairarr:
            write.writerow(items[1].split(','))
        file.close()


# Remove existing directory for fresh result
def ifexitdir(dirarr, filearr):
    for dir in dirarr:
        if os.path.exists(dir):
            shutil.rmtree(dir)
            print("removed "+dir)
    # for file in filearr:
    #     if os.path.exists(file):
    #         os.remove(file)
    #         print("removed " + dir)


def writepred(pred_labels, predlabel_path):
    idx2lb = {}
    for idx, lb in enumerate(pred_labels):
        if lb == -1:
            continue
        idx2lb[idx] = lb
    inst_num = len(pred_labels)
    print('coverage: {} / {} = {:.4f}'.format(len(idx2lb), inst_num, 1. * len(idx2lb) / inst_num))
    write_meta(predlabel_path, idx2lb, inst_num=inst_num)
