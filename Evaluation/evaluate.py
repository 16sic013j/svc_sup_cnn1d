from utils import Timer, np, pairwise



def _read_meta(fn):
    labels = list()
    lb_set = set()
    with open(fn) as f:
        for lb in f.readlines():
            lb = int(lb.strip())
            labels.append(lb)
            lb_set.add(lb)
    return np.array(labels), lb_set


def evaluation(label_file_path, pred_label_path):
    gt_labels, gt_lb_set = _read_meta(label_file_path)
    pred_labels, pred_lb_set = _read_meta(pred_label_path)

    print('#inst: gt({}) vs pred({})'.format(len(gt_labels), len(pred_labels)))
    print('#cls: gt({}) vs pred({})'.format(len(gt_lb_set), len(pred_lb_set)))

    with Timer('evaluate with {}'.format(pairwise)):
        ave_pre, ave_rec, fscore = pairwise(gt_labels, pred_labels)
    print('ave_pre: {:.4f}, ave_rec: {:.4f}, fscore: {:.4f}'.format(ave_pre, ave_rec, fscore))
    return fscore
