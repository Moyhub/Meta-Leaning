import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np

def log_arguments(logger, args, info=None):
    """
    Add arguments to log.
    """
    logger.info(info if info else '==> Current parameters are:')
    if isinstance(args, dict):
        for k in sorted(args.keys()):
            logger.info('\t{}={}'.format(k, args[k]))
    else:
        tmp = '{}'.format(args).split(', ')
        for ele in tmp:
            logger.info('\t{}'.format(ele))

def draw_curve(features, seq_lens, selected_id=8, suffix=''):
    if len(features) != len(seq_lens):
        raise ValueError('==> Number of examples not match! Please check the input...')
    if not os.path.exists('data/feature_id_{}'.format(selected_id)):
        os.mkdir('data/feature_id_{}'.format(selected_id))

    for id, example in enumerate(features):
        plt.plot(example[:, selected_id].tolist())
        plt.xlim((0, 380))
        plt.ylim((0, 1))
        plt.savefig('data/feature_id_{}/rul_{}_id_{}_{}.png'.format(selected_id, seq_lens[id], id, suffix))
        plt.clf()
    plt.close()

def draw_life_cyc_dist(fn, seq_lens_train, seq_lens_test):
    plt.plot(sorted(seq_lens_train), color='red', label='train')
    plt.plot(sorted(seq_lens_test), color='blue', label='test')
    plt.title('distribution of the life cycles')
    plt.legend()
    plt.savefig(fn)
    plt.show()
    plt.close()
    return


def plot_comparison(predictions, target_ruls, seq_lens, fn, scalar=5):
    predictions = np.array(predictions)
    target_ruls = np.array(target_ruls)
    seq_lens = np.array(seq_lens)

    is_positive = target_ruls - predictions >= 0
    gap_positive = scalar * (target_ruls - predictions)[is_positive]
    is_negative = predictions - target_ruls >= 0
    gap_negative = scalar * (predictions - target_ruls)[is_negative]

    fig, ax = plt.subplots()

    ax.scatter(seq_lens[is_positive], target_ruls[is_positive], s=gap_positive, label='lower predictions',
               c='tab:blue', alpha=0.4, edgecolors='none')
    ax.scatter(seq_lens[is_negative], target_ruls[is_negative], s=gap_negative, label='higher predictions',
               c='tab:orange', alpha=0.4, edgecolors='none')

    ax.set_xlabel('Sequence Lengths')
    ax.set_ylabel('Target RULs')
    ax.legend()
    ax.grid(True)
    plt.savefig(fn)
    plt.close()

    return
