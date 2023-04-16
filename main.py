# -- coding:UTF-8 --
import torch, logging, argparse, random, os, json, math, time
import numpy as np
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter

from utils import log_arguments, plot_comparison
from data_loader import load_raw_examples, Corpus
from data_loader import get_cri
from modeling import Model, Learner


# from optimizer import WarmupLinearSchedule, AdamW

def init_logger(level='INFO', file_handler=True, log_fn=None):
    logger = logging.getLogger()
    logger.setLevel(level)
    BASIC_FORMAT = "%(asctime)s:%(levelname)s:%(message)s"
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
    chlr = logging.StreamHandler()  # 输出到控制台的handler
    chlr.setFormatter(formatter)
    chlr.setLevel(level)  # 也可以不设置，不设置就默认用logger的level
    logger.addHandler(chlr)
    if file_handler:
        log_path = '{}/log'.format(args.model_dir) if not log_fn else log_fn  # if log_fn不为空，执行前面那句，否则log_path = log_fn
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        if args.do_train:
            log_file = '{}/data-{}-datatest-{}_supportSize-{}_innerSteps-{}_lrMeta-{}_lrInner-{}_seed-{}.txt'\
                .format(log_path,args.data_fn,args.datatest_fn,args.support_size, args.inner_steps, args.lr_meta, args.lr_inner, args.seed) #改文件名
        
        if args.do_eval:
        # 1.2
            log_file = '{}/data-{}-datatest-{}_supportSize-{}_innerSteps-{}_lrInner-{}_seed-{}.txt'\
                .format(log_path,args.data_fn,args.datatest_fn,args.support_size, args.inner_steps , args.lr_inner, args.seed) ## 改文件名，此时的保存方式不是按时间命名而是按参数
        fhlr = logging.FileHandler(log_file)  # 输出到文件的handler
        fhlr.setFormatter(formatter)
        logger.addHandler(fhlr)
    logger.info('Finish setting logger...')

    return logger


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpu_id >= 0:
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True


def compute_RMSE(logger, args, mode, predictions, ground_true, input_lens, desc=None):
    logger.info("############### Compute RMSEs @ mode [%s] ###############", mode)
    logger.info("\tNum examples = %d", len(predictions))
    ## compute RMSE
    if len(predictions) != len(ground_true) or len(predictions) != len(input_lens):
        raise ValueError('Invalid input! -- moy')
    MSE = 0
    p_ruls, g_ruls, gaps = [], [], []
    with open('{}/result_{}_{}.txt'.format(args.model_dir, mode, desc), 'w', encoding='utf-8') as fw:
        fw.write('gaps\tp_ruls\tg_ruls\n')
        for pred, grdt, curr_len in zip(predictions, ground_true, input_lens):
            p_rul, g_rul = curr_len / pred - curr_len, curr_len / grdt - curr_len
            gap = p_rul - g_rul
            MSE += gap * gap
            p_ruls.append(p_rul)
            g_ruls.append(g_rul)
            gaps.append(gap)
            fw.write('{}\t{}\t{}\n'.format(gap, p_rul, g_rul))
    RMSE = math.sqrt(MSE / len(predictions))
    logger.info("\tRMSE = %.4f", RMSE)
    return RMSE


def train_meta(args, corpus_train, corpus_valid, corpus_test):
    tb_writer = SummaryWriter(log_dir='{}/log'.format(args.model_dir))
    t_total_meta = corpus_train.n_total * args.n_epochs // args.train_batch_size
    t_total_inner = t_total_meta * args.inner_steps
    logger.info('=============== Scheme: Meta Learning ===============')
    logger.info("\tNum examples = %d", corpus_train.n_total)
    logger.info("\tNum epochs = %d", args.n_epochs)
    logger.info("\tBatch size = %d", args.train_batch_size)
    logger.info("\tTotal meta optimization steps = %d", t_total_meta)
    logger.info("\tTotal inner optimization steps = %d", t_total_inner)

    learner = Learner(logger, args, corpus_train.d_input, t_total_meta, t_total_inner)

    global_step = 0
    train_loss, log_loss = 0.0, 0.0
    valid_RMSEs, test_RMSEs = [], []
    output_RMSE = None
    for epoch_i in range(args.n_epochs):
        for batch_i in range(corpus_train.n_total // args.train_batch_size):
            batch_query, batch_support = corpus_train.get_batch_meta(args.train_batch_size, args.device)  # Get batch数据，在train时也会得到reference
            # batch_query包含input_seq,pred,pred_mask,padding_mask等
            loss, lr = learner.forward_meta(batch_query, batch_support)

            train_loss += loss
            if global_step % args.logging_steps == 0:
                tb_writer.add_scalar("train_loss", (train_loss - log_loss) / args.logging_steps,
                                     global_step)  # add_scalar将一个标量加入到summary中
                logger.info("Epoch: {}\t global_step: {}/{}\t lr: {:.5f}\t loss: {:.4f}".format(epoch_i, global_step,
                                                                                                t_total_meta, lr, (
                                                                                                            train_loss - log_loss) / args.logging_steps))
                log_loss = train_loss

            global_step += 1

        # evaluate every epoch
        predictions, ground_true, test_loss = learner.evaluate_meta(corpus_test, args.device)
        test_RMSE = compute_RMSE(logger, args, 'TEST', predictions, ground_true, corpus_test.query_lens,
                                 desc=str(global_step))
        tb_writer.add_scalar("RMSE/test", test_RMSE, epoch_i)
        tb_writer.add_scalar("Loss/test", test_loss, epoch_i)

        predictions, ground_true, valid_loss = learner.evaluate_meta(corpus_valid, args.device)
        valid_RMSE = compute_RMSE(logger, args, 'VALID', predictions, ground_true, corpus_valid.query_lens,
                                  desc=str(global_step))
        tb_writer.add_scalar("RMSE/valid", valid_RMSE, epoch_i)
        tb_writer.add_scalar("Loss/valid", valid_loss, epoch_i)
        if len(valid_RMSEs) == 0 or valid_RMSE < min(valid_RMSEs):
            output_test_RMSE = test_RMSE
            logger.info('==> Minimal valid RMSE!')
            # save model
            logger.info('Save model to %s...', args.model_dir)
            torch.save(learner.model.state_dict(), os.path.join(args.model_dir, "model.bin"))

        test_RMSEs.append(test_RMSE)
        valid_RMSEs.append(valid_RMSE)

    tb_writer.close()
    return output_test_RMSE, valid_RMSEs, test_RMSEs


def evaluate_trained_model(args, corpus_test):
    logger.info('=============== Scheme: Meta Evaluation By Loading Exist Model ===============')
    t_total_meta = corpus_test.n_total * args.n_epochs // args.train_batch_size
    # t_total_inner = t_total_meta * args.inner_steps
    learner = Learner(logger, args, corpus_test.d_input, t_total_meta, None)

    if args.support_size > 0:
        predictions, ground_true, test_loss = learner.evaluate_meta(corpus_test, args.device)
    else:
        predictions, ground_true, test_loss = learner.evaluate_NOmeta(corpus_test, args.device)

    test_RMSE = compute_RMSE(logger, args, 'TEST', predictions, ground_true, corpus_test.query_lens, desc=str(1))


def train_NOmeta(args, corpus_train, corpus_valid, corpus_test):
    tb_writer = SummaryWriter(log_dir='{}/log'.format(args.model_dir))
    t_total_meta = corpus_train.n_total * args.n_epochs // args.train_batch_size
    # t_total_inner = t_total_meta * args.inner_steps
    logger.info('=============== Scheme: Normal Learning ===============')
    logger.info("\tNum examples = %d", corpus_train.n_total)
    logger.info("\tNum epochs = %d", args.n_epochs)
    logger.info("\tBatch size = %d", args.train_batch_size)
    logger.info("\tTotal optimization steps = %d", t_total_meta)

    learner = Learner(logger, args, corpus_train.d_input, t_total_meta, t_total_inner=None)

    global_step = 0
    train_loss, log_loss = 0.0, 0.0
    valid_RMSEs, test_RMSEs = [], []
    output_RMSE = None
    for epoch_i in range(args.n_epochs):
        for batch_i in range(corpus_train.n_total // args.train_batch_size):
            batch_query, _ = corpus_train.get_batch_meta(args.train_batch_size, args.device)
            loss, lr = learner.forward_NOmeta(batch_query)

            train_loss += loss
            if global_step % args.logging_steps == 0:
                tb_writer.add_scalar("train_loss", (train_loss - log_loss) / args.logging_steps,
                                     global_step) 
                logger.info("Epoch: {}\t global_step: {}/{}\t lr: {:.5f}\t loss: {:.4f}".format(epoch_i, global_step,
                                                                                                t_total_meta, lr, (
                                                                                                            train_loss - log_loss) / args.logging_steps))
                log_loss = train_loss

            global_step += 1

        # evaluate every epoch
        predictions, ground_true, test_loss = learner.evaluate_NOmeta(corpus_test, args.device)
        test_RMSE = compute_RMSE(logger, args, 'TEST', predictions, ground_true, corpus_test.query_lens,
                                 desc=str(global_step))
        tb_writer.add_scalar("RMSE/test", test_RMSE, epoch_i)
        tb_writer.add_scalar("Loss/test", test_loss, epoch_i)

        predictions, ground_true, valid_loss = learner.evaluate_NOmeta(corpus_valid, args.device)
        valid_RMSE = compute_RMSE(logger, args, 'VALID', predictions, ground_true, corpus_valid.query_lens,
                                  desc=str(global_step))
        tb_writer.add_scalar("RMSE/valid", valid_RMSE, epoch_i)
        tb_writer.add_scalar("Loss/valid", valid_loss, epoch_i)
        if len(valid_RMSEs) == 0 or valid_RMSE < min(valid_RMSEs):
            output_test_RMSE = test_RMSE
            logger.info('==> Minimal valid RMSE!')
            # save model
            logger.info('Save model to %s...', args.model_dir)
            torch.save(learner.model.state_dict(), os.path.join(args.model_dir, "model.bin"))

        test_RMSEs.append(test_RMSE)
        valid_RMSEs.append(valid_RMSE)

    tb_writer.close()
    return output_test_RMSE, valid_RMSEs, test_RMSEs


def main(args):
    # Read raw examples from data file. 这里是读取数据,数据都是list，包含Tensor数据 run1.sh中的数据全是3，便是我们读取3中的数据进行reference，train_data便是reference
    # 这里的数据还没有进行切割，train_data[0].shape = [192,16] 
    train_data, test_data, target_ruls = load_raw_examples(logger, args.train_data_fn, args.test_data_fn,
                                                           args.target_ruls_fn, args.filter_kernel_size, selected_id=-1)

    if args.support_size > 0:
        weights, _, _ = get_cri(logger, train_data)
    else:
        weights = None

    ## Build train/test/valid corpus. 这个位置是构建Corpus，这里的命名是为了保存数据。
    data_suffix = 'data-{}-datatest-{}_aug-{}_noise-{}_supportSize-{}_seed-{}'.format(args.data_fn, args.datatest_fn,
                                                                                      args.aug_ratio,
                                                                                      args.noise_amplitude,
                                                                                      args.support_size, args.seed)
    if args.do_train:
        # train data # 根据args.override_data_cache决定是否重新构建数据
        if args.override_data_cache or not os.path.exists('data/train_{}'.format(data_suffix)):
            corpus_train = Corpus(logger, args, train_data, 'TRAIN', support_size=args.support_size, weights=weights,
                                  references=train_data, grdt_ruls=None)
            torch.save(corpus_train, 'data/train_{}'.format(data_suffix))
        else:
            corpus_train = torch.load('data/train_{}'.format(data_suffix))
        # valid data
        if args.override_data_cache or not os.path.exists('data/valid_{}'.format(data_suffix)):
            corpus_valid = Corpus(logger, args, train_data, 'VALID', support_size=args.support_size, weights=weights,
                                  references=train_data, grdt_ruls=None)
            torch.save(corpus_valid, 'data/valid_{}'.format(data_suffix))
        else:
            corpus_valid = torch.load('data/valid_{}'.format(data_suffix))
    # test data
    if args.override_data_cache or not os.path.exists('data/test_{}'.format(data_suffix)):
        corpus_test = Corpus(logger, args, test_data, 'TEST', support_size=args.support_size, weights=weights,
                             references=train_data, grdt_ruls=target_ruls)
        torch.save(corpus_test, 'data/test_{}'.format(data_suffix))
    else:
        corpus_test = torch.load('data/test_{}'.format(data_suffix))

    ## Training model. 
    # if args.do_train:
    #     if args.support_size > 0:  # support_size > 0我们是使用Meta-Learning的方法进行
    #         output_test_RMSE, valid_RMSEs, test_RMSEs = train_meta(args, corpus_train, corpus_valid, corpus_test)
    #     else:
    #         output_test_RMSE, valid_RMSEs, test_RMSEs = train_NOmeta(args, corpus_train, corpus_valid, corpus_test)

    #     logger.info("\tOutput TEST RMSE:\t{:.4f}".format(output_test_RMSE))
    #     logger.info("\tVALID RMSEs:\t{}".format('\t'.join(['{:.4f}'.format(i) for i in valid_RMSEs])))
    #     logger.info("\tTEST RMSEs:\t{}".format('\t'.join(['{:.4f}'.format(i) for i in test_RMSEs])))
    ## Evaluate model.
    if args.do_eval:
        evaluate_trained_model(args, corpus_test)


# 这里是运行环境的初始化
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # load data and data pre-processing
    parser.add_argument('--model_dir', default='', type=str)
    parser.add_argument('--data_fn', default=3, type=int)  # ===>
    parser.add_argument("--datatest_fn", default=3, type=int)
    parser.add_argument('--filter_kernel_size', default=1, type=int, help='window size for moving average.')
    parser.add_argument('--override_data_cache', action='store_true')  # action="store_true"一旦传参便是True

    # data augmentation
    parser.add_argument('--maxRUL', default=125, type=int)  # ===>
    parser.add_argument('--low_ratio', default=0.10, type=int)  # ===>
    parser.add_argument('--high_ratio', default=0.99, type=int)  # ===>
    parser.add_argument('--aug_ratio', default=150, type=int)  # ===>
    parser.add_argument('--noise_amplitude', default=0.01, type=float)  # ===>

    # model choice
    parser.add_argument("--modeltype",default="transformer",type=str,choices=["transformer","lstm","cnn2d","cnn1d"])

    # model
    parser.add_argument('--max_seq_len', default=550, type=int)  # ===> FD001: 365
    parser.add_argument('--d_model', default=128, type=int)
    parser.add_argument('--p_dropout', default=0.1, type=float, help='dropout rate of position encoding')
    parser.add_argument('--n_head', default=4, type=int, help='number of multi-head attention')
    parser.add_argument('--n_layer', default=2, type=int, help='number of Transformer encoder layers')
    parser.add_argument('--dim_feedforward', default=512, type=int,
                        help='dimension of feedforward layer in Transformer')
    parser.add_argument('--e_dropout', default=0.1, type=float, help='dropout rate of Transformer encoder layers')
    parser.add_argument('--activation', default='relu', type=str,
                        help='activation function used in Transformer: relu/gelu')
    parser.add_argument('--layer_norm', action='store_true', help='whether to use layer normalization')

    # meta-learning
    parser.add_argument('--support_size', default=2, type=int)
    parser.add_argument('--inner_steps', default=1, type=int)
    parser.add_argument('--lr_inner', default=0.0001, type=float)
    parser.add_argument('--lr_meta', default=0.001, type=float)

    # training
    parser.add_argument('--n_epochs', default=5, type=int)
    parser.add_argument('--train_batch_size', default=20, type=int)
    parser.add_argument('--eval_batch_size', default=1, type=int)
    parser.add_argument('--lr', default=0.001, type=float)  # ===>
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--warmup_ratio', default=0., type=float)  # ===>
    parser.add_argument('--max_grad_norm', default=5.0, type=float)
    parser.add_argument('--logging_steps', default=50, type=int)

    # other
    parser.add_argument('--seed', default=667, type=int)  # ===>
    parser.add_argument('--gpu_id', default=1, type=int)  #
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_eval', action='store_true')

    args = parser.parse_args()
    # setup data file
    args.train_data_fn = 'data/train_FD00{}.txt'.format(args.data_fn)
    args.test_data_fn = 'data/test_FD00{}.txt'.format(args.datatest_fn)
    args.target_ruls_fn = 'data/RUL_FD00{}.txt'.format(args.datatest_fn)
    # setup random seed
    set_seed(args)
    ## setup GPU
    if args.gpu_id >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
    # setup model directory
    if not os.path.exists(args.modeltype + '_models'):
        os.mkdir(args.modeltype + '_models')


    # debug Setting
    args.override_data_cache = True
    args.do_train = False
    # args.modeltype = "cnn1d"
    # args.support_size = 0
    args.do_eval = True

    if len(args.model_dir) == 0:
        if args.do_eval:
            # 注意如果是基础模型加meta-learning  的fune-tune，supportsize不能设置为0. （论文结果需要注意）这里是Basemodel直接测试
            # args.model_dir = 'data-{}_n_epochs-{}_aug-{}_noise-{}_supportSize-{}_innerSteps-{}_lrMeta-{}_lrInner-{}_warmUp-{}_seed-{}'.format(
            #         1, args.n_epochs, args.aug_ratio, args.noise_amplitude,
            #         args.support_size, args.inner_steps, args.lr_meta, args.lr_inner, args.warmup_ratio, args.seed)\
            args.model_dir = 'data-{}_n_epochs-{}_aug-{}_noise-{}_supportSize-{}_innerSteps-{}_lrMeta-{}_lrInner-{}_warmUp-{}_seed-{}'.format(
                    1, args.n_epochs, args.aug_ratio, args.noise_amplitude,
                    0, args.inner_steps, args.lr_meta, args.lr_inner, args.warmup_ratio, args.seed)
            # 这里是basemodel finetune在测试
            # args.model_dir = 'data-{}_n_epochs-{}_aug-{}_noise-{}_supportSize-{}_innerSteps-{}_lrMeta-{}_lrInner-{}_warmUp-{}_seed-{}'.format(
            #         1, args.n_epochs, args.aug_ratio, args.noise_amplitude,
            #         0, args.inner_steps, args.lr_meta, args.lr_inner, args.warmup_ratio, args.seed)
        if args.do_train:
            args.model_dir = 'data-{}_n_epochs-{}_aug-{}_noise-{}_supportSize-{}_innerSteps-{}_lrMeta-{}_lrInner-{}_warmUp-{}_seed-{}'.format(
                args.data_fn, args.n_epochs, args.aug_ratio, args.noise_amplitude,
                args.support_size, args.inner_steps, args.lr_meta,args.lr_inner, args.warmup_ratio, args.seed)

    args.model_dir = os.path.join(args.modeltype+'_models', args.model_dir)
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)
    # setup logger 设置日志输出
    logger = init_logger(level='INFO', file_handler=True)
    log_arguments(logger, args, '==> Training/Evaluation parameters are:')
    # dump args
    logger.info('Dump arguments to %s...', args.model_dir)
    torch.save(args, os.path.join(args.model_dir, "args.bin"))

    # debug Setting
    # args.override_data_cache = True
    # args.do_train = False
    # # args.modeltype = "cnn1d"
    # # args.support_size = 0
    # args.do_eval = True

    main(args)
