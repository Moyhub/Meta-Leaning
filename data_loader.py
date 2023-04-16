import torch, random, heapq, numpy
import torch.nn.functional as F
from utils import draw_life_cyc_dist, draw_curve
from dtw import *

def read_data_(fn):
    # param:
    #   fn: file to read data.
    # return:
    #   data: tensor, n_seq * [time_steps * n_feature].
    #   seq_lens: list, length of each time sequence.
    idxs = [0, 1, 4, 5, 6, 9, 10, 11, 13, 14, 15, 16, 17, 19, 22, 23]  # 13

    # selected features
    curr_example_id = 1
    data = []

    features = []
    with open(fn, 'r', encoding='utf-8') as reader:
        for line in reader:
            line = line.strip()
            if len(line) > 0:
                line = line.split()
                id = int(line[0])
                # time_step = int(line[1])
                if id == curr_example_id:
                    features.append([float(i) for i in line[2:]])
                else:
                    data.append(torch.tensor(features)[:, idxs])
                    features = []
                    curr_example_id = id

                    features.append([float(i) for i in line[2:]])
    if len(features) > 0:
        data.append(torch.tensor(features)[:, idxs])

    total_T = 0
    max_T = 0
    min_T = 666
    for d in data:
        total_T += d.size(0)
        max_T = max(max_T, d.size(0))
        min_T = min(min_T, d.size(0))
    mean_T = total_T / len(data)
    print(mean_T, max_T, min_T)
    return data  #len=100,其中的数据长度不同

def read_data(logger, fn, idxs):
    # param:
    #   fn: file to read data.
    # return:
    #   data: tensor, n_seq * [time_steps * n_feature].
    #   seq_lens: list, length of each time sequence.

    # selected features
    logger.info('==> Read data from {}...'.format(fn))
    logger.info('\tThe selected feature idxs are: {}'.format(', '.join([str(i) for i in idxs])))

    curr_example_id = 1
    data = []

    features = []
    with open(fn, 'r', encoding='utf-8') as reader:
        for line in reader:
            line = line.strip()
            if len(line) > 0:
                line = line.split()
                id = int(line[0])
                # time_step = int(line[1])
                if id == curr_example_id:
                    features.append([float(i) for i in line[2:]])
                else:
                    data.append(torch.tensor(features)[:, idxs])
                    features = []
                    curr_example_id = id

                    features.append([float(i) for i in line[2:]])
    if len(features) > 0:
        data.append(torch.tensor(features)[:, idxs])

    return data  #len=100,其中的数据长度不同

def read_rul(logger, fn):
    logger.info('==> Read RULsfrom {}...'.format(fn))
    ruls = []
    with open(fn, 'r', encoding='utf-8') as reader:
        for line in reader:
            line = line.strip()
            if len(line) > 0:
                ruls.append(int(line))
    logger.info("\tmin_rul: {}, max_rul: {}".format(min(ruls), max(ruls)))
    return ruls

def min_max_normalization(logger, data, min_val=None, max_val=None):
    """
    :param data: A list of examples, each example is a tensof of shape (seq_len, n_feature).
    :return:
        A list of normalized data with the same shape.
    """
    logger.info('==> Min_max normalization...')
    features = torch.cat(data) # seq_len*N x n_feature
    if min_val is None:
        min_val, _ = torch.min(features, dim=0)
        logger.info('\tThe min value is {}'.format(min_val.tolist()))
    else:
        logger.info('\tWith given min value {}'.format(min_val.tolist()))
        for ele in data: # Rescale the data points that are smaller than min_val to min_val.
            mask = ele < min_val
            ele[mask] = min_val.repeat(ele.shape[0], 1)[mask]
    if max_val is None:
        max_val, _ = torch.max(features, dim=0)
        logger.info('\tThe max value is {}'.format(max_val.tolist()))
    else:
        logger.info('\tWith given max value {}'.format(max_val.tolist()))
        for ele in data: # Rescale the data points that are larger than max_val to max_val.
            mask = ele > max_val
            ele[mask] = max_val.repeat(ele.shape[0], 1)[mask]
    data = [(ele - min_val) / (max_val - min_val) for ele in data]
    return data, min_val, max_val

def filtering(logger, data, kernel_size):
    # input:
    #   data: list of examples, each example is a tensor of `seq_len x feature_dim`
    #   kernel_sive: window size for moving average.
    # output:
    #   ans: list of examples, each example is tensor of `(seq_len-kernel_size+1) x feature_dim`
    logger.info('==> Average filtering (kernal_size = {})...'.format(kernel_size))
    ans = []

    filter = torch.tensor([[1.] * kernel_size]) / kernel_size # in_channels x kernel_size, in_channels = 1
    filter = filter.unsqueeze(0) # out_channels = 1, out_channels x in_channels x kernel_size
    padding = kernel_size - 1 # Used in Causal Convolution
    for ele in data: # ele: seq_len x feature_dim
        ele = ele.transpose(0, 1) # feature_dim x seq_len
        ele = ele.unsqueeze(1) # in_channels = 1, feature_dim x in_channels x seq_len
        # tmp = F.conv1d(ele, filter) # 0 padding, feature_dim(minibatch) x out_channels x seq_len
        tmp = F.conv1d(ele, filter, padding=padding)[:, :, :-padding]
        tmp = tmp.squeeze(1) # feature_dim(minibatch) x seq_len'
        tmp = tmp.transpose(0, 1) # seq_len x feature_dim
        ans.append(tmp)

    return ans

def get_seq_lens(data):
    res = []
    for example in data:
        res.append(example.size(0))
    return res

def load_raw_examples(logger, fn_training, fn_test, fn_ruls, filter_kernel_size=1, selected_id=-1):
    """
    :param filter_kernel_size: window size for moving average, 1d.
    :param selected_id: to plot the monitored values of the selected dimension for all examples.
    :return:
        A list of examples. Each example is represented by a tensor of (seq_len, n_feature).
        training data, test data and its corresponding RULs.
    """
    idxs = [0, 1, 4, 5, 6, 9, 10, 11, 13, 14, 15, 16, 17, 19, 22, 23]  # 13
    # idxs = [4, 5, 6, 9, 10, 11, 13, 14, 15, 16, 17, 19, 22, 23]
    # idxs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]  # 13


    # read data for training
    data_train = read_data(logger, fn_training, idxs)
    if filter_kernel_size > 1:
        data_train = filtering(logger, data_train, kernel_size=filter_kernel_size)
    data_train, min_val, max_val = min_max_normalization(logger, data_train)
    if selected_id >= 0:
        draw_curve(data_train, get_seq_lens(data_train), selected_id=selected_id, suffix='train_kernel_size_{}'.format(filter_kernel_size))

    # read data for testing
    data_test = read_data(logger, fn_test, idxs)
    if filter_kernel_size > 1:
        data_test = filtering(logger, data_test, kernel_size=filter_kernel_size)
    input_lens = get_seq_lens(data_test) # 获取测试数据长度
    ruls = read_rul(logger, fn_ruls)
    seq_lens_test = [i + j for i, j in zip(input_lens, ruls)] #这里是将测试集的剩余寿命长度和输入序列长度相加，即总寿命
    input_ratio = [i / j for i, j in zip(input_lens, seq_lens_test)] # 百分比
    logger.info('==> Input length ratio of the [TEST] data:')
    logger.info('\tmin_ratio = %.4f', min(input_ratio))
    logger.info('\tmax_ratio = %.4f', max(input_ratio))
    data_test, _, _ = min_max_normalization(logger, data_test, min_val, max_val)
    if selected_id >= 0:
        draw_curve(data_test, seq_lens_test, selected_id=selected_id, suffix='test_kernel_size_{}'.format(filter_kernel_size))

    # draw_life_cyc_dist('data/dist_of_life_cyc.png', seq_lens_train, seq_lens_test)

    if len(data_test) != len(ruls):
        raise ValueError('Invalid input data! --- wuqh')

    return data_train, data_test, ruls

# 这个位置是学姐当时那篇论文构建的指标，可以认为是一个权重
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
        mono += torch.true_divide(torch.abs(torch.sum(dF > 1e-4, dim=1) - torch.sum(dF < -1e-4, dim=1)) , (T-1))

    corr, mono = corr / n_examples, mono / n_examples
    cri = corr + mono
    cri = cri / torch.sum(cri)
    tmp = [str(i) for i in cri.tolist()]
    logger.info('\tThe weights are: {}'.format(', '.join(tmp)))
    return cri, corr, mono

class InputFeatures(object):
    def __init__(self, input_seq, padding_mask, pred, pred_mask):
        self.input_seq = input_seq
        self.padding_mask = padding_mask
        self.pred = pred
        self.pred_mask = pred_mask

class Corpus(object):
    def __init__(self, logger, args, examples, mode, support_size=-1, weights=None, references=None, grdt_ruls=None):
        ## data
        self.query_features = []
        self.query_lens = []
        self.support_features = [] # active when support_size > 0
        self.weights = weights # active when support_size > 0
        self.support_size = support_size
        self.d_input = examples[0].size(-1) # 输入数据维度

        ## 超参I
        self.max_seq_len = args.max_seq_len # 550
        self.max_rul = args.maxRUL # 125
        # for data augmentation
        self.aug_ratio = args.aug_ratio # 150
        self.low_ratio = args.low_ratio 
        self.noise_amplitude = args.noise_amplitude

        ## sampler
        self.mode = mode
        self.n_total = self.initialize_corpus(logger, examples, grdt_ruls, references)
        self.batch_start_idx = 0
        if self.mode == 'TRAIN':
            self.batch_idxs = numpy.random.permutation(self.n_total) #返回一个随机排列的数列
        else:
            self.batch_idxs = numpy.array([i for i in range(self.n_total)])


    def initialize_corpus(self, logger, examples, grdt_ruls, references):
        logger.info('==> Initialize [{}] Corpus...'.format(self.mode))

        if self.mode == 'TEST':
            if grdt_ruls is None or len(examples) != len(grdt_ruls):
                raise ValueError('Invalid input! -- wuqh')
            cnt = 0
            for exam, rul in zip(examples, grdt_ruls): # exam: seq_len x d_input
                # if cnt % 1 == 0:
                #     logger.info(cnt)
                cnt += 1
                self._append_converted_examples_to_features(exam, rul, references)

        elif self.mode == 'TRAIN':
            for exam_raw in examples:
                seq_len, _ = exam_raw.size()
                for _ in range(self.aug_ratio):  # 每一个example切150个
                # for cut_idx in range(int(seq_len * self.low_ratio), seq_len):
                    cut_idx = random.randint(int(seq_len * self.low_ratio), seq_len) #随机切分,最小从0.1开始
                    rul = seq_len - cut_idx
                    exam = exam_raw[:cut_idx, :] + self.noise_amplitude * (torch.rand((cut_idx, self.d_input)) - 0.5) #加噪声和切分数据
                    self._append_converted_examples_to_features(exam, rul, references)

        elif self.mode == 'VALID':
            for exam_raw in examples:
                seq_len, _ = exam_raw.size()
                cut_idx = random.randint(int(seq_len * self.low_ratio), seq_len)
                rul = seq_len-cut_idx
                exam = exam_raw[:cut_idx, :] + self.noise_amplitude * (torch.rand((cut_idx, self.d_input)) - 0.5)
                self._append_converted_examples_to_features(exam, rul, references)
        else:
            raise ValueError('Invalid input! -- wuqh')
        
        n_total = len(self.query_features)
        logger.info('\tNumber of examples:  {}'.format(n_total))
        return n_total

    def _append_converted_examples_to_features(self, input_exam, rul, references):
        input_features, input_len = self.__convert_example_to_features(input_exam, rul)
        self.query_features.append(input_features)
        self.query_lens.append(input_len)
        if self.support_size > 0:
            support_set = []
            # for idx in self.__get_similar_ids_l2(input_exam, references): #这个位置便是寻找similar features. # input_exam.shape=[45,16]
            # for idx in self.__get_similar_ids_random(len(references)):
            for idx in self.__get_similar_ids_dtw(input_exam,references):
                input_exam_sup = references[idx][:input_len, :] #切割到相同的长度
                rul_sup = max(references[idx].size(0) - input_len,0)
                sup_features, _ = self.__convert_example_to_features(input_exam_sup, rul_sup)
                support_set.append(sup_features)
            self.support_features.append(support_set)

    def __convert_example_to_features(self, input_seq, rul):
        input_len, d_input = input_seq.size()

        paddings = torch.zeros(self.max_seq_len-input_len, d_input)
        input_seq = torch.cat((input_seq, paddings), dim=0) # max_seq_len x d_input,这里是全部填充到550的长度
        padding_mask = [0] * input_len + [1] * (self.max_seq_len - input_len)
        padding_mask = torch.tensor(padding_mask, dtype=torch.bool) # max_seq_len,长度550   0000---1111

        pred_mask = torch.zeros(self.max_seq_len, dtype=torch.bool) # 注意 padding 和 rul 的 mask 是反的
        pred_mask[input_len-1] = True # max_seq_len #只有input_len - 1 这个位置为True
        if self.mode == 'TRAIN':
            rul = min(rul, self.max_rul)
        pred = input_len / (input_len + rul) # scalor,输入数据长度占总长度的百分比


        input_features = InputFeatures(input_seq, padding_mask, pred, pred_mask)
        return input_features, input_len

    def __get_similar_ids_l2(self, query, references):
        scores = []
        heapq.heapify(scores)

        for idx, ref in enumerate(references):
            # if len(ref) < len(query):
            #     continue
            min_len = min(len(ref), len(query))
            dis = torch.mean(torch.abs(ref[:min_len] - query[:min_len]) ** 2 * self.weights)

            if len(scores) < self.support_size:
                heapq.heappush(scores, (-dis, idx)) # 最大堆,heappush是将元素以树形结构存储，子节点大于父节点，兄弟节点不会排序
            else:
                if dis < -scores[0][0]: #不断将树中最大的元素（因为排序以-排序，因此是倒叙根节点弹出）
                    heapq.heappop(scores)
                    heapq.heappush(scores, (-dis, idx))

        similar_idxs = []
        while scores:
            _, idx = heapq.heappop(scores)
            similar_idxs.append(idx)

        return similar_idxs #获得最相似的几个元素的index

    def __get_similar_ids_dtw(self,query,references): #后续继续跑
        scores = []
        heapq.heapify(scores)
        for idx,ref in enumerate(references):
            dis = 0
            min_len = min(len(ref), len(query))
            for dim_index in range(query.size()[-1]):
                dis += dtw(query.numpy()[:min_len,dim_index],ref.numpy()[:int(min_len*1.4),dim_index],keep_internals=True,open_begin=False,open_end=True).distance * self.weights[dim_index]
                #alignment.plot(type="twoway",offset=-0.4)

            if len(scores) < self.support_size:
                heapq.heappush(scores, (-dis, idx))  # 最大堆,heappush是将元素以树形结构存储，子节点大于父节点，兄弟节点不会排序
            else:
                if dis < -scores[0][0]:  # 不断将树中最大的元素（因为排序以-排序，因此是倒叙根节点弹出）
                    heapq.heappop(scores)
                    heapq.heappush(scores, (-dis, idx))

        similar_idxs = []
        while scores:
            _, idx = heapq.heappop(scores)
            similar_idxs.append(idx)
        print("Simliar index",similar_idxs)
        return similar_idxs  # 获得最相似的几个元素的index

    def __get_similar_ids_random(self,N):
        # 随机选择
        similar_idxs = random.choices(range(N), k=self.support_size)

        return similar_idxs


    def reset_batch_info(self):
        print('!!! Reset batch info !!! mode: [{}]'.format(self.mode))
        self.batch_start_idx = 0
        if self.mode == 'TRAIN': # shuffle
            self.batch_idxs = numpy.random.permutation(self.n_total)
        else:
            self.batch_idxs = numpy.array([i for i in range(self.n_total)])

    def get_batch_meta(self, batch_size, device):
        if self.batch_start_idx + batch_size > self.n_total: # Note that '>=' doesn't work here.
            self.reset_batch_info()
        
        query_batch, support_batch = [], []
        start_id = self.batch_start_idx
        for i in range(start_id, start_id + batch_size):
            idx = self.batch_idxs[i]
            # Build query set.
            query_i = self.query_features[idx]
            query_item = {
                'input_seq': query_i.input_seq.unsqueeze(0).to(device), # 1 x seq_len x d_model
                'padding_mask': query_i.padding_mask.unsqueeze(0).to(device), # 1 x seq_len
                'pred': torch.tensor(query_i.pred).unsqueeze(0).to(device), # scalor
                'pred_mask': query_i.pred_mask.unsqueeze(0).to(device) # 1 x seq_len
            }
            query_batch.append(query_item)
            # Build support set.
            if self.support_size > 0:
                support_i = self.support_features[idx]
                support_item = {
                    'input_seq': torch.stack([f.input_seq for f in support_i]).to(device), # support_size x seq_len x d_model
                    'padding_mask': torch.stack([f.padding_mask for f in support_i]).to(device), # support_size x seq_len
                    'pred': torch.tensor([f.pred for f in support_i]).to(device), # support_size
                    'pred_mask': torch.stack([f.pred_mask for f in support_i]).to(device) # # support_size x seq_len
                } 
                support_batch.append(support_item)

        self.batch_start_idx += batch_size

        return query_batch, support_batch
    
    # def get_batches(self, batch_size, device="cuda"):
    #     # raise ValueError('Illegal entrance! -- wuqh')
    #     batches = []

    #     if self.mode == 'TRAIN': # shuffle
    #         idxs = np.random.permutation(self.n_total)
    #         features = [self.query_features[i] for i in idxs]
    #     else:
    #         features = self.query_features

    #     for i in range(0, self.n_total, batch_size):
    #         batch_features = features[i : min(self.n_total, i + batch_size)]
    #         batch = {
    #             'input_seq': torch.stack([f.input_seq for f in batch_features]).to(device),
    #             'padding_mask': torch.stack([f.padding_mask for f in batch_features]).to(device),
    #             'pred': torch.tensor([f.pred for f in batch_features]).to(device),
    #             'pred_mask': torch.stack([f.pred_mask for f in batch_features]).to(device)
    #         }
    #         batches.append(batch)

    #     return batches

if __name__ == '__main__':
    read_data_('data/train_FD001.txt')
    read_data_('data/test_FD003.txt')