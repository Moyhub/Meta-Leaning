import torch, math, copy, numpy
import torch.nn as nn
from optimizer import WarmupLinearSchedule, AdamW
# import torch.nn.functional as F
# from torch.utils.data import SequentialSampler, DataLoader

class Embeddings(nn.Module):
    """
    https://github.com/harvardnlp/annotated-transformer/blob/master/The%20Annotated%20Transformer.ipynb
    """
    def __init__(self, d_input, d_model, p_dropout, max_len):
        super(Embeddings, self).__init__()
        self.mapping = nn.Linear(d_input, d_model)
        self.dropout = nn.Dropout(p_dropout)
        # self.d_model = d_model

        ## wuqh
        # pe = torch.zeros(max_len, d_model)
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1)
        # more stable to compute the positional encodings in log space.
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # 1 x max_len x d_model
        self.register_buffer('pe', pe)

    def forward(self, input):
        """
        :param input: batch_size x seq_len x d_input
        :return:
        """
        embeds = self.mapping(input) # batch_size x seq_len x d_model
        # x = x + self.pe[:, :x.size(1)]
        output = embeds + self.pe
        return self.dropout(output)

class LSTModel(nn.Module):
    def __init__(self,support_size,train_batch_size):
        super(LSTModel,self).__init__()

        self.lstmlayer1 = nn.LSTM(input_size = 16, hidden_size = 32, num_layers=1, batch_first=True)
        self.lstmlayer2 = nn.LSTM(input_size = 32, hidden_size = 64, num_layers=1, batch_first=True)
        self.nnlayer1 = nn.Linear(64,8)
        self.nnlayer2 = nn.Linear(8,8)
        self.nnlayer3 = nn.Linear(8,1)
        self.sigmoid = torch.nn.Sigmoid()
        if support_size > 0:
            batch_size = support_size
        else:
            batch_size = train_batch_size
    
    def forward(self, input, padding_mask, pred_mask, preds=None):
        output,(hn,cn) = self.lstmlayer1(input)
        output,(hn,cn) = self.lstmlayer2(output)
        out = self.nnlayer1(output)
        out = self.nnlayer2(out)  # batch_size*max_len*dim
        out = out[pred_mask] # batch_size * dim

        out = self.sigmoid(self.nnlayer3(out)).squeeze(-1) # batch_size
        if preds is not None:
            loss_fn = nn.MSELoss()
            loss = loss_fn(out, preds)
        else:
            loss = None
        return out, loss

class CNN1d(nn.Module):
    def __init__(self, max_seq, in_size=10,dilate=[1,1,1,1],TimeDim=9,Featuredim=1):
        super().__init__()
        self.linear_size = 100
        self.in_features = 16
        self.seq_len = max_seq
        self.conv1 = self.conv(1, in_size, ks=[TimeDim, Featuredim], dila=[dilate[0], dilate[0]], pad = [(TimeDim-1)*dilate[0]//2,(Featuredim-1)*dilate[0]//2])
        self.conv2 = self.conv(in_size, in_size, ks=[TimeDim, Featuredim], dila=[dilate[1], dilate[1]], pad = [((TimeDim-1)*dilate[1])//2,((Featuredim-1)*dilate[1])//2])
        self.conv3 = self.conv(in_size, in_size, ks=[TimeDim, Featuredim], dila=[dilate[2], dilate[2]], pad = [((TimeDim-1)*dilate[2])//2,((Featuredim-1)*dilate[2])//2])
        self.conv4 = self.conv(in_size, in_size, ks=[TimeDim, Featuredim], dila=[dilate[3], dilate[3]], pad =[((TimeDim-1)*dilate[3])//2,((Featuredim-1)*dilate[3])//2])
        self.conv5 = self.conv(in_size, 1, ks=[3,1], dila=[1, 1], pad = [(3-1)//2,(1-1)//2])
        self.dropout = nn.Dropout(0.5)      
        self.fc_1 = self.fc(self.in_features*self.seq_len, self.linear_size, activation=True)
        self.fc_2 = self.fc(self.linear_size, 1)
        self.sigmoid = torch.nn.Sigmoid()
    
    def conv(self,c_in, c_out, ks, dila, sd=1, pad=[1, 0]):
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=ks, stride=sd, padding=pad, dilation=dila, bias=False),
            nn.Tanh(),
        )
    def fc(self, c_in, c_out, activation=False):
        if activation:
            return nn.Sequential(
                nn.Linear(c_in, c_out),
                nn.Tanh(),
            )
        else:
            return nn.Linear(c_in, c_out)

    def forward(self, input, padding_mask, pred_mask, preds=None):
        batch_size, seq_len, dim = tuple(input.size()) # batch_size(meta-learning inner update:batch_size = support_size) * max_len * dim
        x = input.view(batch_size, 1, seq_len, dim)
        x = self.conv1(x)                   # batch_size x cnn_out_channels=10 x max_len x dim  # padding项 将卷积导致的尺寸缩小抵消 batch_size * 
        x = self.conv2(x)                 
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)                   # batch_size x cnn_out_channels=1 x seq_len x in_features=D1
        x = x.view(x.size(0), -1)           # batch_size  x seq_len x in_features=D1
        x = self.dropout(x)
        x = self.fc_1(x)
        out = self.sigmoid(self.fc_2(x)).squeeze(-1)

        if preds is not None:
            loss_fn = nn.MSELoss()
            loss = loss_fn(out, preds)
        else:
            loss = None
        return out, loss


class CNN2d(nn.Module):
    def __init__(self, max_seq, in_size=10,dilate=[1,2,4,8],TimeDim=7,Featuredim=5):
        super().__init__()
        self.linear_size = 100
        self.in_features = 16
        self.seq_len = max_seq
        self.conv1 = self.conv(1, in_size, ks=[TimeDim, Featuredim], dila=[dilate[0], dilate[0]], pad = [(TimeDim-1)*dilate[0]//2,(Featuredim-1)*dilate[0]//2])
        self.conv2 = self.conv(in_size, in_size, ks=[TimeDim, Featuredim], dila=[dilate[1], dilate[1]], pad = [((TimeDim-1)*dilate[1])//2,((Featuredim-1)*dilate[1])//2])
        self.conv3 = self.conv(in_size, in_size, ks=[TimeDim, Featuredim], dila=[dilate[2], dilate[2]], pad = [((TimeDim-1)*dilate[2])//2,((Featuredim-1)*dilate[2])//2])
        self.conv4 = self.conv(in_size, in_size, ks=[TimeDim, Featuredim], dila=[dilate[3], dilate[3]], pad =[((TimeDim-1)*dilate[3])//2,((Featuredim-1)*dilate[3])//2])
        self.conv5 = self.conv(in_size, 1, ks=[3,3], dila=[1, 1], pad = [(3-1)//2,(3-1)//2])
        self.dropout = nn.Dropout(0.5)      
        self.fc_1 = self.fc(self.in_features*self.seq_len, self.linear_size, activation=True)
        self.fc_2 = self.fc(self.linear_size, 1)
        self.sigmoid = torch.nn.Sigmoid()
    
    def conv(self,c_in, c_out, ks, dila, sd=1, pad=[1, 0]):
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=ks, stride=sd, padding=pad, dilation=dila, bias=False),
            nn.ReLU(),
        )
    def fc(self, c_in, c_out, activation=False):
        if activation:
            return nn.Sequential(
                nn.Linear(c_in, c_out),
                nn.ReLU(),
            )
        else:
            return nn.Linear(c_in, c_out)

    def forward(self, input, padding_mask, pred_mask, preds=None):
        batch_size, seq_len, dim = tuple(input.size()) # batch_size(meta-learning inner update:batch_size = support_size) * max_len * dim
        x = input.view(batch_size, 1, seq_len, dim)
        x = self.conv1(x)                   # batch_size x cnn_out_channels=10 x max_len x dim  # padding项 将卷积导致的尺寸缩小抵消 batch_size * 
        x = self.conv2(x)                 
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)                   # batch_size x cnn_out_channels=1 x seq_len x in_features=D1
        x = x.view(x.size(0), -1)           # batch_size  x seq_len x in_features=D1
        x = self.dropout(x)
        x = self.fc_1(x)
        out = self.sigmoid(self.fc_2(x)).squeeze(-1)

        if preds is not None:
            loss_fn = nn.MSELoss()
            loss = loss_fn(out, preds)
        else:
            loss = None
        return out, loss

class Model(nn.Module):
    def __init__(self, d_input, d_model, p_dropout, max_seq_len,
                 n_head, n_layer, dim_feedforward, e_dropout, activation, layer_norm):
        super(Model, self).__init__()
        self.d_model = d_model

        self.embed = Embeddings(d_input, d_model, p_dropout, max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward=dim_feedforward, dropout=e_dropout, activation=activation)
        if layer_norm:
            layer_norm = nn.LayerNorm([max_seq_len, d_model])
        else:
            layer_norm = None
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layer, norm=layer_norm)

        self.projection = nn.Linear(d_model, 1)
        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, input, padding_mask, pred_mask, preds=None):
        """
        :param input: batch_size x max_seq_len x d_input
        :param padding_mask: batch_size x max_seq_len, the mask for the src keys per batch. provides specified elements in the key to be ignored by the attention.
        :param pred_mask: batch_size x max_seq_len
        :param preds: batch_size
        :return:
        """
        embeds = self.embed(input) # batch_size x max_seq_len x d_model
        embeds = embeds.transpose(0, 1) # max_seq_len x batch_size x d_model
        logits = self.encoder(embeds, mask=None, src_key_padding_mask=padding_mask) # max_seq_len x batch_size x d_model
        logits = logits.transpose(0, 1) # batch_size x max_seq_len x d_model
        logits = logits.reshape(-1, self.d_model) # batch_size * max_seq_len x d_model
        pred_mask = pred_mask.view(-1) # batch_size * max_seq_len
        logits = logits[pred_mask] # batch_size x d_model

        output = self.sigmoid(self.projection(logits)).squeeze(-1) # batch_size

        if preds is not None:
            loss_fn = nn.MSELoss()
            loss = loss_fn(output, preds)
        else:
            loss = None

        return output, loss

class CausalConv1d(torch.nn.Conv1d):
    # copied from https://github.com/pytorch/pytorch/issues/1333, by arogozhnikov
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        self.__padding = (kernel_size - 1) * dilation

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[:, :, :-self.__padding]
        return result

class Learner(nn.Module):
    def __init__(self, logger, args, d_input, t_total_meta, t_total_inner):
        super(Learner, self).__init__()

        if args.modeltype == "transformer":
            self.model = Model(d_input, args.d_model, args.p_dropout, args.max_seq_len, args.n_head, args.n_layer,
                        args.dim_feedforward, args.e_dropout, args.activation, layer_norm=args.layer_norm)
        elif args.modeltype == "lstm":
            self.model = LSTModel(args.support_size,args.train_batch_size)
        elif args.modeltype == "cnn1d":
            self.model = CNN1d(max_seq = args.max_seq_len)
        elif args.modeltype == "cnn2d":
            self.model = CNN2d(max_seq = args.max_seq_len)
        else:
            raise Exception('Please provide the right network')

        if args.do_eval:
            logger.info('Load model from %s...', args.model_dir)
            self.model.load_state_dict(torch.load('{}/model.bin'.format(args.model_dir)))
        if args.gpu_id >= 0:
            self.model.to(args.device)

        opt_params = self.get_optimizer_grouped_parameters(logger, args.weight_decay)
        self.opt_meta = AdamW(opt_params, lr=args.lr_meta, eps=1e-8, weight_decay=args.weight_decay)
        self.scheduler_meta = WarmupLinearSchedule(self.opt_meta, warmup_steps=int(t_total_meta * args.warmup_ratio), t_total=t_total_meta)
        self.opt_inner = AdamW(opt_params, lr=args.lr_inner, eps=1e-8, weight_decay=args.weight_decay)
        # self.scheduler_inner = WarmupLinearSchedule(self.opt_inner, warmup_steps=int(t_total_inner * args.warmup_ratio), t_total=t_total_inner)
        self.inner_steps = args.inner_steps
        self.max_grad_norm = args.max_grad_norm

        
    def get_optimizer_grouped_parameters(self, logger, weight_decay):
        logger.info('==> Group parameters for optimization...')
        logger.info('    Parameters to update are:')
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                assert False, "parameters to update with requires_grad=False"
            else:
                logger.info('\t{}'.format(n))

        no_decay = ["bias", "norm1.weight", "norm2.weight"]
        outputs = [
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": weight_decay},
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0}
        ]
        return outputs

    def get_names(self):
        names = [n for n, p in self.model.named_parameters() if p.requires_grad]
        return names

    def get_params(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        return params
    
    def load_weights(self, names, params):
        model_params = self.model.state_dict()
        for n, p in zip(names, params):
            model_params[n].data.copy_(p.data)
    
    def load_gradients(self, names, grads):
        model_params = self.model.state_dict(keep_vars=True)
        for n, g in zip(names, grads):
            model_params[n].grad.data.add_(g.data) # accumulate

    def inner_update(self, data_support):
        self.model.train()
        for i in range(self.inner_steps):
            self.opt_inner.zero_grad()
            _, loss = self.model.forward(data_support['input_seq'], data_support['padding_mask'], data_support['pred_mask'], data_support['pred'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.opt_inner.step()
            # self.scheduler_inner.step()
        return loss.item()

    def forward_meta(self, batch_query, batch_support):
        names = self.get_names()
        params = self.get_params()
        weights = copy.deepcopy(params)

        meta_grad, meta_loss = [], []
        # compute meta_grad of each task
        for task_id in range(len(batch_query)):
            self.inner_update(batch_support[task_id])
            data_query = batch_query[task_id]
            _, loss = self.model.forward(data_query['input_seq'], data_query['padding_mask'], data_query['pred_mask'], data_query['pred'])
            grad = torch.autograd.grad(loss, params, allow_unused=True)
            meta_grad.append(grad)
            meta_loss.append(loss.item())

            self.load_weights(names, weights)
        
        # accumulate grads of all tasks to param.grad
        self.opt_meta.zero_grad()

        # similar to backward()
        for g in meta_grad:
            self.load_gradients(names, g)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.opt_meta.step()
        self.scheduler_meta.step()

        ave_loss = numpy.mean(numpy.array(meta_loss))

        return ave_loss, self.scheduler_meta.get_lr()[0]
    
    def forward_NOmeta(self, batch):
        input_seq = torch.cat([f['input_seq'] for f in batch])
        padding_mask = torch.cat([f['padding_mask'] for f in batch])
        pred_mask = torch.cat([f['pred_mask'] for f in batch])
        pred = torch.cat([f['pred'] for f in batch])

        self.model.train()
        self.opt_meta.zero_grad()
        _, loss = self.model.forward(input_seq, padding_mask, pred_mask, pred)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.opt_meta.step()
        self.scheduler_meta.step()

        return loss.item(), self.scheduler_meta.get_lr()[0]

    def evaluate_meta(self, corpus, device):
        names = self.get_names()
        params = self.get_params()
        weights = copy.deepcopy(params)

        preds, grdts, losses = [], [], []
        for item_id in range(corpus.n_total):
            # train on support examples
            eval_query, eval_support = corpus.get_batch_meta(batch_size=1, device=device)
            self.inner_update(eval_support[0])
            # eval on pseudo query examples (test examples)
            self.model.eval()
            data_query = eval_query[0]
            with torch.no_grad():
                pred, loss = self.model.forward(data_query['input_seq'], data_query['padding_mask'], data_query['pred_mask'], data_query['pred'])
            
            preds.append(pred.detach().cpu().item())
            grdts.append(data_query['pred'].to('cpu').item())
            losses.append(loss.detach().cpu().item())

            self.load_weights(names, weights)

        return preds, grdts, numpy.mean(numpy.array(losses))
    
    def evaluate_NOmeta(self, corpus, device):
        preds, grdts, losses = [], [], []
        self.model.eval()
        for item_id in range(corpus.n_total):
            eval_query, _ = corpus.get_batch_meta(batch_size=1, device=device)
            data_query = eval_query[0]
            with torch.no_grad():
                pred, loss = self.model.forward(data_query['input_seq'], data_query['padding_mask'], data_query['pred_mask'], data_query['pred'])
            
            preds.append(pred.detach().cpu().item())
            grdts.append(data_query['pred'].to('cpu').item())
            losses.append(loss.detach().cpu().item())
        
        return preds, grdts, numpy.mean(numpy.array(losses))
    
    # def evaluate_NOmeta(self, corpus, device):
    #     # raise ValueError('Illegal entrance! -- wuqh')
    #     data_batches = corpus.get_batches(batch_size=1, device=device)
    #     self.model.eval()

    #     preds, grdts, losses = [], [], []
    #     for batch in data_batches:
    #         with torch.no_grad():
    #             pred, loss = self.model.forward(batch['input_seq'], batch['padding_mask'], batch['pred_mask'], batch['pred'])
            
    #         preds.extend(pred.detach().cpu().tolist())
    #         grdts.extend(batch['pred'].to('cpu').tolist())
    #         losses.append(loss.detach().cpu().item())

    #     return preds, grdts, numpy.mean(numpy.array(losses))