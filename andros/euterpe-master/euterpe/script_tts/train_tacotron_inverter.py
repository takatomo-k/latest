import os
import sys

import argparse
import json, yaml
import numpy as np
import itertools
import time
import timeit
import operator
import pickle
import tabulate as tab
from tqdm import tqdm
import pandas as pd
from tensorboardX import SummaryWriter

def tqdm_wrapper(obj) :
    return tqdm(obj, ascii=True, ncols=50)

# pytorch #
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torchev.utils.serializer import ModelSerializer
from torchev.nn.modules.loss import MaskedCrossEntropyLoss
from torchev.utils.mask_util import generate_seq_mask
from torchev.utils.helper import torchauto, tensorauto

# utilbox #
from utilbox.data_util import iter_minibatches, iter_minibucket, iter_minibucket_block
from utilbox.math_util import assert_nan
from utilbox.parse_util import all_config_load
from utilbox.log_util import logger_stdout_file
from euterpe.data.data_generator import group_feat_timestep, feat_sil_from_stat
from euterpe.data.data_iterator import DataIterator, TextIterator
from euterpe.common.loader import DataLoader, LoaderDataType
from euterpe.common.batch_data import batch_speech

DEBUG = False
def parse() :
    parser = argparse.ArgumentParser(description='training script for Tacotron 2')

    # Tacotron V2 (new Tacotron model from Google)
    # format file V2 : model_cfg is defined by file
    parser.add_argument('--model_cfg', type=all_config_load, help='model configuration file')

    # data_cfg using new type npz loader #
    parser.add_argument('--data_in_cfg', type=str, required=True, help='data in (prediction from Tacotron core model')
    parser.add_argument('--data_out_cfg', type=str, required=True, help='data out (ground truth linear spectrogram')

    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--batchsize', type=int, default=20)
    parser.add_argument('--loss', type=str, choices=['L1', 'L2'], default=['L2'], nargs='+')
    parser.add_argument('--loss_freq_cfg', type=json.loads, default={'topn':0.25, 'coeff':1}, help='loss additional for raw spectrogram from 0-25 % freq range')
    parser.add_argument('--mask_dec', action='store_true')
    parser.add_argument('--group', type=int, default=1, 
            help='group n-frame acoustic togather for 1 time step decoding (reduce number of step)')
    parser.add_argument('--pad_sil', type=int, default=2)

    parser.add_argument('--opt', type=str, default='Adam')
    parser.add_argument('--lrate', type=float, default=0.001) # default is not always converge
    parser.add_argument('--decay', type=float, default=1.0, help='decay lrate after no dev cost improvement')
    parser.add_argument('--grad_clip', type=float, default=5.0) # grad clip to prevent NaN
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='L2 weight decay regularization')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--result', type=str, required=True, help='model result path')
    parser.add_argument('--bucket', action='store_true')
    parser.add_argument('--block', type=int, default=-1)
    parser.add_argument('--sortagrad', type=int, default=-1)

    parser.add_argument('--cutoff', type=int, default=-1, help='cutoff frame larger than x')

    parser.add_argument('--update_part', type=str, default='all', help='freeze/unfreeze parameter')

    parser.add_argument('--model_pt', type=str, default=None, help='use pre-trained model for initialization')

    parser.add_argument('--chkpoint_interval', type=int, default=2, help='save model every x epoch')
    parser.add_argument('--mem', action='store_true', default=False)

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
        
    group = opts['group']
    # dataset #
    data_in_cfg = yaml.load(open(opts['data_in_cfg']))
    data_out_cfg = yaml.load(open(opts['data_out_cfg']))

    feat_in_iterator = DataLoader.load_feat(data_in_cfg, 
            data_type=LoaderDataType.TYPE_NP, in_memory=opts['mem'])

    feat_out_iterator = DataLoader.load_feat(data_out_cfg, 
            data_type=LoaderDataType.TYPE_NP, in_memory=opts['mem'])

    feat_in_stat = pickle.load(open(data_in_cfg['feat']['stat'], 'rb'))
    feat_in_sil = feat_sil_from_stat(feat_in_stat)

    feat_out_stat = pickle.load(open(data_out_cfg['feat']['stat'], 'rb'))
    feat_out_sil = feat_sil_from_stat(feat_out_stat)

    print("Finish loading dataset ...") 
    
    NDIM_IN = feat_in_iterator['train'].get_feat_dim() * group
    NDIM_OUT = feat_out_iterator['train'].get_feat_dim() * group

    if opts['model_pt'] is not None :
        model = ModelSerializer.load_config(os.path.join(opts['model_pt'], 'model.cfg')) 
        model.load_state_dict(torch.load(os.path.join(opts['model_pt'], 'model.mdl')))
        print('[info] load pretrained model')
    else : 
        _model_cfg = opts['model_cfg']
        _model_cfg['in_size'] = NDIM_IN
        _model_cfg['out_size'] = NDIM_OUT
        model = ModelSerializer.load_config(_model_cfg)

    # l1 or l2 # 
    def criterion(input, target, mask, size_average=True) :
        loss = 0
        if 'L1' in opts['loss'] :
            loss += torch.abs(input - target)
        if 'L2' in opts['loss'] :
            loss += (input - target)**2 # batch x len x ndim #

        loss = torch.mean(loss, 2) # batch x len #
        if opts['mask_dec'] :
            loss = loss * mask
        loss = torch.sum(loss) # sum all rest dim #
        if size_average :
            loss /= input.size(0)
        return loss

    def criterion_freq(input, target, mask, size_average=True) :
        """
        aux loss for prioritize optimizing loss on lower frequency
        """
        if opts['loss_freq_cfg'] is None :
            return 0
        assert 0 < opts['loss_freq_cfg']['topn'] <= 1
        assert opts['loss_freq_cfg']['coeff'] > 0
        ndim = int(input.size(-1) * opts['loss_freq_cfg']['topn'])
        loss = criterion(input[:, :, 0:ndim], target[:, :, 0:ndim], mask, size_average=True)
        return opts['loss_freq_cfg']['coeff'] * loss

    if opts['gpu'] >= 0 :
        model.cuda(opts['gpu'])
     
    # setting optimizer #
    list_opt_param = []
    if opts['update_part'] == 'all' :
        list_opt_param += list(model.parameters())
    else :
        raise NotImplementedError()

    opt = getattr(torch.optim, opts['opt'])(list_opt_param, lr=opts['lrate'], weight_decay=opts['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5, min_lr=5e-5)

    def fn_batch(feat_in_mat, feat_in_len, feat_out_mat, feat_out_len, train_step=True) :

        feat_in_mat = Variable(feat_in_mat)
        feat_out_mat = Variable(feat_out_mat)

        feat_in_mask = Variable(generate_seq_mask(feat_in_len, opts['gpu']))
        feat_out_mask = Variable(generate_seq_mask(feat_out_len, opts['gpu']))
        model.train(train_step)
        pred_mat_output = model(feat_in_mat, feat_in_len)

        # main : loss mel spectrogram #
        loss_core = criterion(pred_mat_output, feat_out_mat, feat_out_mask)
        
        # optional : aux loss for lower frequency #
        loss_core_freq = criterion_freq(pred_mat_output, feat_out_mat, feat_out_mask)

        loss_feat = loss_core + loss_core_freq

        # combine all loss #
        loss = loss_feat

        if train_step :
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), opts['grad_clip'])
            opt.step()

            # write report #
            tf_writer.add_scalar('loss', loss.data[0], global_step=tf_writer._n_iter)
            tf_writer._n_iter += 1
        return loss.data.sum()
    
    N_TRAIN_SIZE = len(feat_in_iterator['train'].key)
    N_DEV_SIZE = len(feat_in_iterator['dev'].key)
    N_TEST_SIZE = len(feat_in_iterator['test'].key)
    train_len = feat_in_iterator['train'].get_feat_length()
    dev_len = feat_in_iterator['dev'].get_feat_length()
    test_len = feat_in_iterator['test'].get_feat_length()

    #### SORT BY LENGTH - SortaGrad ####
    sorted_train_idx = np.argsort(feat_in_iterator['train'].get_feat_length()).tolist()
    sorted_dev_idx = np.argsort(feat_in_iterator['dev'].get_feat_length()).tolist()
    sorted_test_idx = np.argsort(feat_in_iterator['test'].get_feat_length()).tolist()
    
    sorted_train_len = operator.itemgetter(*sorted_train_idx)(feat_in_iterator['train'].get_feat_length())
    sorted_dev_len = operator.itemgetter(*sorted_dev_idx)(feat_in_iterator['dev'].get_feat_length())
    sorted_test_len = operator.itemgetter(*sorted_test_idx)(feat_in_iterator['test'].get_feat_length())
    
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
            raise ValueError("Error : folder & data already existed !!!")

    logger = logger_stdout_file(os.path.join(opts['result'], 'report.log'))
    tf_writer = SummaryWriter(log_dir=os.path.join(opts['result'], 'tfboard.event'))
    tf_writer._n_iter = 0

    # exclude cutoff #
    exc_idx = {}
    exc_idx['train'] = set([x for x in range(len(train_len)) if train_len[x] > opts['cutoff']])
    exc_idx['dev'] = set([x for x in range(len(dev_len)) if dev_len[x] > opts['cutoff']])
    exc_idx['test'] = set([x for x in range(len(test_len)) if test_len[x] > opts['cutoff']])
    for set_name in ['train', 'dev', 'test'] :
        logger.info('[pre] exclude set {} : {} utts'.format(set_name, len(exc_idx[set_name])))
    
    print("=====START=====")
    prev_dev_loss = 2**64
    best_dev_loss = 2**64
    best_dev_loss_ep = 0
    for ee in range(1, EPOCHS+1) :
        start_time = time.time()
        
        mloss = dict(train=0, dev=0, test=0)
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
                    shuffle=False if ee <= opts['sortagrad'] else True, 
                    pad=True, excludes=exc_idx['train'])
                dev_rr = iter_minibucket_block(sorted_dev_idx, opts['block'], sorted_dev_len, 
                    shuffle=False, pad=True, excludes=exc_idx['dev'])
                test_rr = iter_minibucket_block(sorted_test_idx, opts['block'], sorted_test_len, 
                    shuffle=False, pad=True, excludes=exc_idx['test'])
        else :
            train_rr = iter_minibatches(sorted_train_idx, BATCHSIZE, 
                    shuffle=False if ee <= opts['sortagrad'] else True, excludes=exc_idx['train'])
            dev_rr = iter_minibatches(sorted_dev_idx, BATCHSIZE, 
                    shuffle=True, excludes=exc_idx['dev'])
            test_rr = iter_minibatches(sorted_test_idx, BATCHSIZE, 
                    shuffle=True, excludes=exc_idx['test'])

        train_rr = [sort_reverse(x, train_len) for x in train_rr] 
        dev_rr = [sort_reverse(x, dev_len) for x in dev_rr] 
        test_rr = [sort_reverse(x, test_len) for x in test_rr] 
        
        def save_model(model_name) :
            torch.save(ModelSerializer.convert_param_to_cpu(model.state_dict()), os.path.join(opts['result'], '{}.mdl'.format(model_name)))
            ModelSerializer.save_config(os.path.join(opts['result'], '{}.cfg'.format(model_name)), model.get_config())
            json.dump(opts, open(os.path.join(opts['result'], 'script.opts'), 'w'), indent=4)

        for set_name, set_rr, set_train_mode in [('train', train_rr, True), ('dev', dev_rr, False), ('test', test_rr, False)] :
            for rr in tqdm_wrapper(set_rr) :
                tic = timeit.default_timer()
                curr_key_list = feat_in_iterator[set_name].get_key_by_index(rr)
                curr_feat_in_list = feat_in_iterator[set_name].get_feat_by_key(curr_key_list)
                curr_feat_out_list = feat_out_iterator[set_name].get_feat_by_key(curr_key_list)
                # print(1, timeit.default_timer() - tic); tic = timeit.default_timer()
                feat_in_mat, feat_in_len = batch_speech(opts['gpu'], curr_feat_in_list,
                        feat_sil=feat_in_sil, group=group, start_sil=0,
                        end_sil=opts['pad_sil'])
                feat_out_mat, feat_out_len = batch_speech(opts['gpu'], curr_feat_out_list,
                        feat_sil=feat_out_sil, group=group, start_sil=0,
                        end_sil=opts['pad_sil'])
                # print(2, timeit.default_timer() - tic); tic = timeit.default_timer()
                _tmp_loss = fn_batch(feat_in_mat, feat_in_len, 
                        feat_out_mat, feat_out_len, 
                        train_step=set_train_mode)
                # print(3, timeit.default_timer() - tic); tic = timeit.default_timer()
                _tmp_count = len(rr)
                assert_nan(_tmp_loss)
                mloss[set_name] += _tmp_loss * _tmp_count
                mcount[set_name] += _tmp_count
            pass

        info_header = ['set', 'loss']
        info_table = []
        logger.info("Epoch %d -- lrate %f -- time %.2fs"%(ee, opts['lrate'], time.time() - start_time))
        for set_name in mloss.keys() :
            mloss[set_name] /= mcount[set_name]
            info_table.append([set_name, mloss[set_name]])
        logger.info('\n'+tab.tabulate(info_table, headers=info_header, floatfmt='.3f', tablefmt='rst'))

        if (ee) % opts['chkpoint_interval'] == 0 :
            # save every n epoch
            save_model('model_e{}'.format(ee))

        if best_dev_loss > mloss['dev'] :
            # save best dev model
            best_dev_loss = mloss['dev']
            best_dev_loss_ep = ee
            save_model('best_model')
            logger.info("\t# get best dev loss ... serialized the model")


        prev_dev_loss = mloss['dev']
        # update scheduler
        scheduler.step(mloss['dev'], ee)
        pass

    logger.info("best dev cost: %f at epoch %d"%(best_dev_loss, best_dev_loss_ep))
    pass
