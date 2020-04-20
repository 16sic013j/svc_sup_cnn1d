import numpy as np


def read_meta(fn_meta, verbose=True):
    lb2idxs = {}
    idx2lb = {}
    with open(fn_meta) as f:
        for idx, x in enumerate(f.readlines()):
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


def read_probs(path, inst_num, feat_dim, verbose=True):
    assert (inst_num > 0 or inst_num == -1) and feat_dim > 0
    count = -1
    if inst_num > 0:
        count = inst_num * feat_dim
    # probs = np.genfromtxt(path, delimiter=',',max_rows=count)
    # print(probs)
    probs = np.load(path)
    # if feat_dim > 1:
    #     probs = probs.reshape(inst_num, feat_dim)
    # if verbose:
    #     print('[{}] shape: {}'.format(path, probs.shape))
    return probs


def l2norm(vec):
    # Takes sqrt( sum( vec[n] * vec[n])) to get L2 norm(elucidian distance)
    linalg = np.linalg.norm(vec, axis=1).reshape(-1, 1)
    vec /= linalg
    return vec


def getnormlize_features(feature_file_path):
    print("normalising feature...")

    # Read feature from feature file(feature that was extracted from doc infer vector)
    features = np.load(feature_file_path)

    # Get inverse l2norm of the vector as faiss requires inverse indexes
    features = l2norm(features)

    return features
