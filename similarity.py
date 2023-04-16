from main import init_logger
import data_loader, heapq, random, torch

import numpy as np
from scipy import signal
from dtw import *

def get_cri(logger, examples):
    """
    :param examples: A list of examples, each example is represented as a tensor of shape (seq_len, n_feature).
    :return:
        (cri, corr, mono), where cri = 0.5*corr + 0.5*mono
    """
    logger.info('==> Computing Criterion...')
    if not examples:
        raise ValueError('Invalid input! --wuqh')
    n_examples, n_features = examples[0].size()
    corr = torch.zeros(n_features)
    mono = torch.zeros(n_features)
    filter = torch.tensor([[-1., 1.0]])  # in_channels x kernel_size, in_channels = 1
    filter = filter.unsqueeze(0)  # out_channels = 1, out_channels x in_channels x kernel_size
    for i, ele in enumerate(examples):
        T = ele.size(0)
        F_ave = torch.mean(ele, dim=0) # n_feature
        F_mat = ele - F_ave # seq_len x n_feature
        t_mat = torch.tensor(range(0, T)).reshape((T, 1)) - T/2 # seq_len x 1
        # compute correlation
        corr += torch.abs(torch.sum(F_mat * t_mat, dim=0)) / torch.sqrt(torch.sum(F_mat * F_mat, dim=0) * torch.sum(t_mat * t_mat, dim=0))
        # compute monotonicity
        ele = ele.transpose(0, 1)  # feature_dim x seq_len
        ele = ele.unsqueeze(1)  # in_channels = 1, feature_dim x in_channels x seq_len
        dF = torch.nn.functional.conv1d(ele, filter)  # 0 padding, feature_dim(minibatch) x out_channels x seq_len
        dF = dF.squeeze(1)  # feature_dim(minibatch) x seq_len'
        # dF = dF.transpose(0, 1)  # seq_len x feature_dim
        mono += torch.abs(torch.sum(dF > 1e-4, dim=1) - torch.sum(dF < -1e-4, dim=1)) / (T-1)

    corr, mono = corr / n_examples, mono / n_examples
    cri = corr + mono
    cri = cri / torch.sum(cri)
    tmp = [str(i) for i in cri.tolist()]
    logger.info('\tThe weights are: {}'.format(', '.join(tmp)))
    return cri, corr, mono

def get_similar_examples(logger, query, reference, ruls, k=2, selected_id=-1, weights=None):
    """
    :param query: a list of queries, each query is represented by a tensor of shape (seq_len, n_features).
    :param reference: a list of references, each reference is represented by a tensor of shape (seq_len, n_features).
    :param k: scalar, for top-k most similar examples.
    :return:
        top-K similar examples for each query.
    """
    if not query or not reference or not ruls or len(query) != len(ruls):
        raise ValueError('Please check the input. --wuqh')
    logger.info('==> Get similar examples: k={}, selected_id={}, {} weights...'.format(k, selected_id, 'WITH' if weights is not None else 'WITHOUT'))
    if selected_id >= 0:
        query = [q[:, selected_id] for q in query]
        reference = [ref[:, selected_id] for ref in reference]
        weights = None
    else:
        if weights is None:
            n_features = query[0].size(1)
            weights = torch.ones(n_features) / n_features

    similar_idxs, gaps = [], []
    # ref_ruls = []

    for q_idx in range(len(query)):
        if q_idx % 10 == 0:
            logger.info('\t\tTo No. {}'.format(q_idx))
        similar_idx, ref_rul, gap = _get_similar_ids_l2(query[q_idx], reference, k, ruls[q_idx], weights)
        similar_idxs.append(similar_idx)
        # ref_ruls.extend(ref_rul)
        gaps.extend(gap)

    # ref_ruls = np.array(ref_ruls)
    # logger.info('  mean(ref_fuls) = {}, var(ref_ruls) = {}'.format(np.average(ref_ruls), np.var(ref_ruls)))
    gaps = np.abs(np.array(gaps))
    print(gaps)
    logger.info('\tmean(gaps) = {}'.format(np.average(gaps)))
    logger.info('\tvar(gaps) = {}'.format(np.var(gaps)))
    return similar_idxs

def _get_similar_ids_dtw(query, reference, k, rul, weights, open_begin=False, open_end=True):
    scores = []
    heapq.heapify(scores)

    for idx, ref in enumerate(reference):
        if len(ref) < len(query):
            continue
        if weights is None:
            alignmentOBE = dtw(query, ref[:len(query)], keep_internals=True, step_pattern=asymmetric, open_begin=open_begin, open_end=open_end)
            dis = alignmentOBE.distance
        else:
            dis = 0
            for i in range(query.size(1)):
                q = query[:, i]
                r = ref[:, i] # ref[:len(query), i]
                alignmentOBE = dtw(q, r[:len(query)], keep_internals=True, step_pattern=asymmetric, open_begin=open_begin, open_end=open_end)
                dis += alignmentOBE.distance * weights[i]

        ref_rul = len(ref) - len(query)
        gap = ref_rul - rul

        if len(scores) < k:
            # 最大堆
            heapq.heappush(scores, (-dis, idx, ref_rul, gap))
        else:
            if dis < -scores[0][0]:
                heapq.heappop(scores)
                heapq.heappush(scores, (-dis, idx, ref_rul, gap))

    simiar_idxs, ref_ruls, gaps = [], [], []
    while scores:
        _, idx, ref_rul, gap = heapq.heappop(scores)
        simiar_idxs.append(idx)
        ref_ruls.append(ref_rul)
        gaps.append(gap)

    return simiar_idxs, ref_ruls, gaps


def _get_similar_ids_l2(query, reference, k, rul, weights):
    scores = []
    heapq.heapify(scores)

    for idx, ref in enumerate(reference):
        if len(ref) < len(query):
            continue
        if weights is None:
            dis = torch.sum(torch.abs(ref[:len(query)] - query) ** 2)
        else:
            dis = torch.sum(torch.abs(ref[:len(query)] - query) ** 2 * weights)

        ref_rul = len(ref) - len(query)
        gap = ref_rul - rul

        if len(scores) < k:
            # 最大堆
            heapq.heappush(scores, (-dis, idx, ref_rul, gap))
        else:
            if dis < -scores[0][0]:
                heapq.heappop(scores)
                heapq.heappush(scores, (-dis, idx, ref_rul, gap))

    simiar_idxs, ref_ruls, gaps = [], [], []
    while scores:
        _, idx, ref_rul, gap = heapq.heappop(scores)
        simiar_idxs.append(idx)
        ref_ruls.append(ref_rul)
        gaps.append(gap)

    return simiar_idxs, ref_ruls, gaps

if __name__ == '__main__':
    logger = init_logger(level='INFO', file_handler = True, log_fn='logs')

    args = {
        'fn_train': 'data/train_FD001.txt',
        'fn_test': 'data/test_FD001.txt',
        'fn_rul': 'data/RUL_FD001.txt',
        'top_k': 2,
        'filter_kernel_size': 5,
        'selected_id': -1
    }
    logger.info('Hyper parameters:')
    for k, v in args.items():
        logger.info('\t{}:\t{}'.format(k, v))

    data_train, data_test, ruls = data_loader.load_raw_examples(logger, args['fn_train'], args['fn_test'], args['fn_rul'], args['filter_kernel_size'], args['selected_id'])
    weights, _, _ = get_cri(logger, data_train)
    get_similar_examples(logger, data_test, data_train, ruls, k=args['top_k'], selected_id=args['selected_id'], weights=weights)


    a = 1
    exit(0)

