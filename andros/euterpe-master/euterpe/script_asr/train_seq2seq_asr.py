import os
import sys

import argparse
import json, yaml
import numpy as np
import itertools
import time
import operator
import tabulate as tab

from tqdm import tqdm
def tqdm_wrapper(obj) :
    return tqdm(obj, ascii=True, ncols=50)

# pytorch #
import torch
from torch import nn
from torch.autograd import Variable
from torchev.utils.serializer import ModelSerializer
from torchev.nn.modules.loss_elementwise import ElementwiseCrossEntropy
from torchev.utils.mask_util import generate_seq_mask
from torchev.utils.helper import tensorauto, torchauto
from torchev.optim.lr_scheduler import ReduceLROnPlateauEv

# utilbox #
from utilbox.data_util import iter_minibatches, iter_minibucket, iter_minibucket_block
from utilbox.math_util import assert_nan
from utilbox.log_util import logger_stdout_file
from utilbox.parse_util import all_config_load
from euterpe.common.loader import DataLoader, LoaderDataType
from euterpe.common.batch_data import batch_speech_text
from euterpe.config import constant
from euterpe.util import train_loop

def parse() :
    parser = argparse.ArgumentParser(description='trainer ASR V2 (npz loader with independent file features)')
    parser.add_argument('--model_cfg', type=all_config_load, required=True)
    parser.add_argument('--data_cfg', type=str, required=True)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--batchsize', type=int, default=20)

    parser.add_argument('--lbl_smooth', type=float, default=0.0, 
            help='label smoothing regularization')

    parser.add_argument('--opt', type=str, default='Adam')
    parser.add_argument('--lrate', type=float, default=0.001) # default is not always converge
    parser.add_argument('--reducelr', type=json.loads, 
            default={'factor':0.5, 'patience':3, 'reset':False},
            help="""
            factor: decay lr * factor\n
            patience: wait x epoch until decay\n 
            reset: replace with new optimizer (only for Adam, Adagrad, etc)""") 
    parser.add_argument('--grad_clip', type=float, default=20.0) # grad clip to prevent NaN
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--result', type=str, required=True, help='add (+) for append suffix when we supply pretraining model')
    parser.add_argument('--bucket', action='store_true')
    parser.add_argument('--block', type=int, default=-1)
    parser.add_argument('--sortagrad', type=int, default=-1)
    parser.add_argument('--mem', action='store_true')
    parser.add_argument('--cutoff', type=int, default=-1, help='cutoff frame larger than x')

    parser.add_argument('--model_pt', type=str, default=None, help='use pre-trained model for initialization')
    parser.add_argument('--save_interval', type=int, default=1, help='save model every x epoch')
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


if __name__ == '__main__' :
    opts = vars(parse())
    print(opts)

    # set default device #
    np.random.seed(opts['seed'])
    torch.manual_seed(opts['seed'])
    if opts['gpu'] >= 0 :
        torch.cuda.set_device(opts['gpu'])
        torch.cuda.manual_seed(opts['seed'])
    
    # dataset #
    data_cfg = yaml.load(open(opts['data_cfg']))
    feat_iterator = DataLoader.load_feat(data_cfg, data_type=LoaderDataType.TYPE_NP, in_memory=opts['mem'])
    text_iterator = DataLoader.load_text(data_cfg)

    print("Finish loading dataset ...") 
    
    NDIM = feat_iterator['train'].get_feat_dim()
    NVOCAB = len(text_iterator['train'].get_map_text2idx())
    if opts['model_pt'] is not None :
        assert opts['model_pt'].endswith('.mdl'), 'model pretrained should have *.mdl file extension'
        _model_state_path = opts['model_pt']
        _model_cfg_path = os.path.splitext(opts['model_pt'])[0]+'.cfg'
        model = ModelSerializer.load_config(_model_cfg_path) 
        model.load_state_dict(torch.load(_model_state_path))

        print('[info] load pretrained model')

        # additional #
        if opts['result'].startswith('+') :
            opts['result'] = os.path.dirname(opts['model_pt'])+opts['result'][1:]
            print('[info] append pretrained folder name to result')
    else :
        _model_cfg = opts['model_cfg']
        _model_cfg['enc_in_size'] = NDIM
        _model_cfg['dec_in_size'] = NVOCAB
        _model_cfg['dec_out_size'] = NVOCAB 
        model = ModelSerializer.load_config(_model_cfg)

    crit_weight = tensorauto(opts['gpu'], torch.ones(NVOCAB))
    crit_weight[constant.PAD] = 0
    crit_weight = Variable(crit_weight, requires_grad=False)
    criterion = ElementwiseCrossEntropy(weight=crit_weight, 
            label_smoothing=opts['lbl_smooth'])

    if opts['gpu'] >= 0 :
        model.cuda(opts['gpu'])
        pass
    
    # setting optimizer #
    opt = getattr(torch.optim, opts['opt'])(model.parameters(), lr=opts['lrate'])
    scheduler = ReduceLROnPlateauEv(opt, factor=opts['reducelr']['factor'], patience=opts['reducelr']['patience'], min_lr=5e-5, verbose=True)

    def fn_batch(feat_mat, feat_len, text_mat, text_len, train_step=True) :
        # TODO
        feat_mat = Variable(feat_mat)
        text_input = Variable(text_mat[:, 0:-1])
        text_output = Variable(text_mat[:, 1:])
        model.reset() 
        model.train(train_step)
        model.encode(feat_mat, feat_len)
        batch, dec_len = text_input.size()
        list_pre_softmax = []
        for ii in range(dec_len) :
            _pre_softmax_ii, _ = model.decode(text_input[:, ii])
            list_pre_softmax.append(_pre_softmax_ii)
            pass
        pre_softmax = torch.stack(list_pre_softmax, 1)
        denominator = Variable(torchauto(model).FloatTensor(text_len)-1)
        # average loss based on individual length #
        loss = criterion(pre_softmax.contiguous().view(batch * dec_len, -1), 
                text_output.contiguous().view(batch * dec_len)).view(batch, dec_len).sum(dim=1) / denominator
        loss = loss.mean()

        acc = torch.max(pre_softmax, 2)[1].data.eq(text_output.data).masked_select(text_output.ne(constant.PAD).data).sum() / denominator.sum()
        if train_step :
            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), opts['grad_clip'])
            opt.step()

        return loss.data.sum(), acc.data.sum()
    
    N_TRAIN_SIZE = len(feat_iterator['train'].key)
    N_DEV_SIZE = len(feat_iterator['dev'].key)
    N_TEST_SIZE = len(feat_iterator['test'].key)
    train_len = feat_iterator['train'].get_feat_length()
    dev_len = feat_iterator['dev'].get_feat_length()
    test_len = feat_iterator['test'].get_feat_length()

    train_text_len, dev_text_len, test_text_len = (text_iterator[set_name].get_text_length() for set_name in ['train', 'dev', 'test'])

    #### SORT BY LENGTH - SortaGrad ####
    sorted_train_idx = np.argsort(feat_iterator['train'].get_feat_length()).tolist()
    sorted_dev_idx = np.argsort(feat_iterator['dev'].get_feat_length()).tolist()
    sorted_test_idx = np.argsort(feat_iterator['test'].get_feat_length()).tolist()
    
    sorted_train_len = operator.itemgetter(*sorted_train_idx)(feat_iterator['train'].get_feat_length())
    sorted_dev_len = operator.itemgetter(*sorted_dev_idx)(feat_iterator['dev'].get_feat_length())
    sorted_test_len = operator.itemgetter(*sorted_test_idx)(feat_iterator['test'].get_feat_length())
    
    def sort_reverse(idx, length) : # for encoder mask must be sorted decreasing #
        return sorted(idx, key=lambda x : length[x], reverse=True)
        pass
    ########################

    EPOCHS = opts['epoch']
    BATCHSIZE = opts['batchsize']
    
    # prepare model folder #
    if not os.path.exists(opts['result']) :
        os.makedirs(opts['result'])
    else :
        if len(os.listdir(opts['result'])) > 0:
            raise ValueError("Error : folder & data already existed !!!\n{}".format(opts['result']))
    
    print("=====START=====")
    prev_dev_loss = 2**64
    best_dev_loss = 2**64
    best_dev_loss_ep = 0
    logger = logger_stdout_file(os.path.join(opts['result'], 'report.log'))
    # save current script command #
    logger.info('{}'.format(' '.join([xi for xi in sys.argv])))

    # exclude cutoff #
    MIN_DUR = 32
    exc_idx = {}
    exc_idx['train'] = set([x for x in range(len(train_len)) if train_len[x] > opts['cutoff'] or train_len[x] < MIN_DUR])
    exc_idx['dev'] = set([x for x in range(len(dev_len)) if dev_len[x] > opts['cutoff'] or dev_len[x] < MIN_DUR])
    exc_idx['test'] = set([x for x in range(len(test_len)) if test_len[x] > opts['cutoff'] or test_len[x] < MIN_DUR])
    for set_name, set_data_len in [('train', N_TRAIN_SIZE), ('dev', N_DEV_SIZE), ('test', N_TEST_SIZE)] :
        logger.info('[pre] exclude set {} : {} utts from total {}'.format(set_name, len(exc_idx[set_name]), set_data_len))
    
    ee = 1
    while ee <= EPOCHS :
        start_time = time.time()
        mloss = dict(train=0, dev=0, test=0)
        macc = dict(train=0, dev=0, test=0)
        mcount = dict(train=0, dev=0, test=0)

        # choose standard training or bucket training #
        if opts['bucket'] :
            if opts['block'] == -1 :
                train_rr = iter_minibucket(sorted_train_idx, BATCHSIZE, 
                    shuffle=False if ee <= opts['sortagrad'] else True, 
                    excludes=exc_idx['train'])
                dev_rr = iter_minibucket(sorted_dev_idx, BATCHSIZE, 
                    shuffle=False, excludes=exc_idx['dev'])
                test_rr = iter_minibucket(sorted_test_idx, BATCHSIZE, 
                    shuffle=False, excludes=exc_idx['test'])
            else :
                train_rr = iter_minibucket_block(sorted_train_idx, opts['block'], sorted_train_len, 
                    shuffle=False if ee <= opts['sortagrad'] else True, pad=True,
                    excludes=exc_idx['train'])
                dev_rr = iter_minibucket_block(sorted_dev_idx, opts['block'], sorted_dev_len, 
                    shuffle=False, pad=True, excludes=exc_idx['dev'])
                test_rr = iter_minibucket_block(sorted_test_idx, opts['block'], sorted_test_len, 
                    shuffle=False, pad=True, excludes=exc_idx['test'])
        else :
            train_rr = iter_minibatches(sorted_train_idx, BATCHSIZE, 
                    shuffle=False if ee <= opts['sortagrad'] else True,
                    excludes=exc_idx['train'])
            dev_rr = iter_minibatches(sorted_dev_idx, BATCHSIZE, shuffle=True,
                    excludes=exc_idx['dev'])
            test_rr = iter_minibatches(sorted_test_idx, BATCHSIZE, shuffle=True,
                    excludes=exc_idx['test'])
        ###############################################
        train_rr = [sort_reverse(x, train_len) for x in train_rr] 
        dev_rr = [sort_reverse(x, dev_len) for x in dev_rr] 
        test_rr = [sort_reverse(x, test_len) for x in test_rr] 

        if opts['debug'] :
            # get longest sequence #
            max_text_len = -1
            max_feat_len = -1
            max_text_rr = None
            max_feat_rr = None
            for tmp_rr in train_rr :
                _curr_max_text = max(operator.itemgetter(*tmp_rr)(train_len))
                _curr_max_feat = max(operator.itemgetter(*tmp_rr)(train_text_len))
                if _curr_max_text > max_text_len :
                    max_text_rr = tmp_rr
                if _curr_max_feat > max_feat_len :
                    max_feat_rr = tmp_rr
            logger.info('[debug] max text len: {}, feat len: {}'.format(max_text_len, max_feat_len))
            train_rr = [max_text_rr, max_feat_rr]
        ###############################################

        def save_model(model_name) :
            torch.save(ModelSerializer.convert_param_to_cpu(model.state_dict()), os.path.join(opts['result'], '{}.mdl'.format(model_name)))
            ModelSerializer.save_config(os.path.join(opts['result'], '{}.cfg'.format(model_name)), model.get_config())
            json.dump(opts, open(os.path.join(opts['result'], 'script.opts'), 'w'), indent=4)

        for set_name, set_rr, set_train_mode in [('train', train_rr, True), ('dev', dev_rr, False), ('test', test_rr, False)] :
            for rr in tqdm_wrapper(set_rr) :
                start = time.time()
                curr_key_list = feat_iterator[set_name].get_key_by_index(rr)
                curr_feat_list = feat_iterator[set_name].get_feat_by_key(curr_key_list)
                curr_label_list = text_iterator[set_name].get_text_by_key(curr_key_list)
                assert (feat_iterator[set_name].get_key_by_index(rr) == text_iterator[set_name].get_key_by_index(rr)), 'key(s) not same'
                # print(1, start-time.time()); start = time.time()
                feat_mat, feat_len, text_mat, text_len = batch_speech_text(opts['gpu'], curr_feat_list, curr_label_list) 
                # print(2, start-time.time()); start = time.time()
                _tmp_loss, _tmp_acc = fn_batch(feat_mat, feat_len, text_mat, text_len, train_step=set_train_mode)
                # print(3, start-time.time()); start = time.time()
                _tmp_count = len(rr)
                assert_nan(_tmp_loss)
                mloss[set_name] += _tmp_loss * _tmp_count
                macc[set_name] += _tmp_acc * _tmp_count
                mcount[set_name] += _tmp_count
            pass
        info_header = ['set', 'loss', 'acc']
        info_table = []

        logger.info("Epoch %d -- lrate %f --time %.2fs"%(ee, opt.param_groups[0]['lr'], time.time() - start_time))
        for set_name in mloss.keys() :
            mloss[set_name] /= mcount[set_name]
            macc[set_name] /= mcount[set_name]
            info_table.append([set_name, mloss[set_name], macc[set_name]])
        logger.info('\n'+tab.tabulate(info_table, headers=info_header, floatfmt='.3f', tablefmt='rst'))

        if (ee) % opts['save_interval'] == 0 :
            # save every n epoch
            save_model('model_e{}'.format(ee))

        # serialized best dev model #
        if best_dev_loss > mloss['dev'] :
            best_dev_loss = mloss['dev']
            best_dev_loss_ep = ee
            save_model('best_model')
            logger.info("\t# get best dev loss ... serialized the model")

        prev_dev_loss = mloss['dev']
        # update scheduler
        _sch_on_trigger = scheduler.step(mloss['dev'], ee)
        if _sch_on_trigger :
            if opts['reducelr'].get('reset', False) :
                opt = getattr(torch.optim, opts['opt'])(model.parameters(), lr=opt.param_groups[0]['lr'])
                scheduler.optimizer = opt
                logger.info('\t# scheduler triggered! reset opt & lr: {:.3e}'.format(opt.param_groups[0]['lr']))
        
        # LAST STEP: increment epoch counter 
        ee += 1
        pass
    logger.info("final best dev loss %f at epoch %d"%(best_dev_loss, best_dev_loss_ep))
    pass

