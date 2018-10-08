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

def tqdm_wrapper(obj) :
    return tqdm(obj, ascii=True, ncols=50)

# pytorch #
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torchev.utils.serializer import ModelSerializer
from torchev.nn.modules.loss_elementwise import ElementwiseCrossEntropy
from torchev.utils.mask_util import generate_seq_mask
from torchev.utils.helper import torchauto, tensorauto

# utilbox #
from utilbox.data_util import iter_minibatches, iter_minibucket, iter_minibucket_block
from utilbox.math_util import assert_nan
from utilbox.log_util import logger_stdout_file
from utilbox.parse_util import all_config_load
from euterpe.data.data_generator import group_feat_timestep, feat_sil_from_stat
from euterpe.data.data_iterator import DataIterator
from euterpe.model_spkrec.deep_speaker import DeepSpeakerCNN
from euterpe.common.loader import DataLoader, LoaderDataType
from euterpe.common.batch_data import batch_speech

DEBUG = False
def parse() :
    parser = argparse.ArgumentParser(description='params')
   
    # V2 : model_cfg is defined by file
    parser.add_argument('--model_cfg', type=all_config_load, help='model configuration file')

    parser.add_argument('--data_cfg', type=str, required=True)

    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--batchsize', type=int, default=20)

    parser.add_argument('--lbl_smooth', type=float, default=0.0, 
            help='label smoothing regularization')

    parser.add_argument('--opt', type=str, default='Adam')
    parser.add_argument('--lrate', type=float, default=0.001) # default is not always converge
    parser.add_argument('--decay', type=float, default=1.0, help='decay lrate after no dev cost improvement')
    parser.add_argument('--grad_clip', type=float, default=20.0) # grad clip to prevent NaN
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--result', type=str, required=True)
    # batching strategy
    parser.add_argument('--bucket', action='store_true')
    parser.add_argument('--block', type=int, default=-1)
    parser.add_argument('--sortagrad', type=int, default=-1)

    parser.add_argument('--cutoff', type=int, default=-1, help='cutoff frame larger than x')

    parser.add_argument('--model_pt', type=str, default=None, help='use pre-trained model for initialization')
    parser.add_argument('--chkpoint_interval', type=int, default=2, help='save model every x epoch')
    parser.add_argument('--mem', action='store_true', default=False)
    
    parser.add_argument('--coeff_ce', type=float, default=1)
    parser.add_argument('--coeff_margin', type=float, default=1)

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

    feat_iterator = DataLoader.load_feat(data_cfg, 
            data_type=LoaderDataType.TYPE_NP, in_memory=opts['mem'])

    feat_stat = pickle.load(open(data_cfg['feat']['stat'], 'rb'))
    feat_sil = feat_sil_from_stat(feat_stat)
    data_info = pd.read_csv(data_cfg['misc']['info'], sep=',')
    map_key2spk = dict((str(k), str(v)) for k, v in yaml.load(open(data_cfg['misc']['key2spk'])).items())
    # filter out non-existent speaker #
    available_data_spk = set([map_key2spk[x] for x in feat_iterator['train'].get_key()])
    # assert if all dev & test speakers are covered in training set
    assert available_data_spk >= set([map_key2spk[x] for x in feat_iterator['dev'].get_key()])
    assert available_data_spk >= set([map_key2spk[x] for x in feat_iterator['test'].get_key()])
    available_data_spk = sorted(list(available_data_spk))
    map_spk2id = dict((y, x) for x, y in enumerate(available_data_spk))
    
    print("... Finish loading dataset ...") 
    
    NDIM = feat_iterator['train'].get_feat_dim()
    NSPEAKER = len(map_spk2id)

    if opts['model_pt'] is not None :
        model = ModelSerializer.load_config(os.path.join(opts['model_pt'], 'model.cfg')) 
        model.load_state_dict(torch.load(os.path.join(opts['model_pt'], 'model.mdl')))
        print('[info] load pretrained model')
    else : 
        # num_speaker is optional for pre-train w/ softmax 
        model = DeepSpeakerCNN(in_size=NDIM, num_speaker=NSPEAKER, **opts['model_cfg'])

    def criterion_ce(input, target, size_average=True) :
        # loss = F.cross_entropy(input, target, size_average=size_average)
        loss = ElementwiseCrossEntropy(label_smoothing=opts['lbl_smooth'])(input, target)
        loss = loss.sum() / (input.shape[0] if size_average else 1)
        return loss

    def criterion_tripletmargin(input, positive, negative, margin=0.1, size_average=True) :
        loss = F.triplet_margin_loss(input, positive, negative, margin=margin)
        if not size_average :
            return loss * input.size(0)
        return loss

    if opts['gpu'] >= 0 :
        model.cuda(opts['gpu'])
        pass
    
    # setting optimizer #
    list_opt_param = []
    list_opt_param += list(model.parameters())

    opt = getattr(torch.optim, opts['opt'])(list_opt_param, lr=opts['lrate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, min_lr=5e-5, verbose=True)
    def fn_batch_ce(feat_mat, feat_len, speaker_list, train_step=True) :
        feat_mat = Variable(feat_mat)
        feat_mask = Variable(generate_seq_mask([x for x in feat_len], opts['gpu']))
        speaker_list_id = [map_spk2id[x] for x in speaker_list]
        speaker_list_id = Variable(torchauto(model).LongTensor(speaker_list_id))

        model.reset()
        model.train(train_step)
        batch, dec_len, _ = feat_mat.size()

        pred_emb = model(feat_mat, feat_len)
        pred_softmax = model.forward_softmax(pred_emb)

        loss = criterion_ce(pred_softmax, speaker_list_id) * opts['coeff_ce']
        acc = torch.max(pred_softmax, 1)[1].data.eq(speaker_list_id.data).sum() / batch

        if train_step :
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), opts['grad_clip'])
            opt.step()
        return loss.data.sum(), acc

    def fn_batch_triplet() :
        raise NotImplementedError()

    N_TRAIN_SIZE = len(feat_iterator['train'].key)
    N_DEV_SIZE = len(feat_iterator['dev'].key)
    N_TEST_SIZE = len(feat_iterator['test'].key)
    train_len = feat_iterator['train'].get_feat_length()
    dev_len = feat_iterator['dev'].get_feat_length()
    test_len = feat_iterator['test'].get_feat_length()

    
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
            raise ValueError("Error : folder & data already existed !!!")

    logger = logger_stdout_file(os.path.join(opts['result'], 'report.log'))

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
    best_dev_loss_epo = 0
    for ee in range(1, EPOCHS+1) :
        start_time = time.time()
        
        mloss = dict(train=0, dev=0, test=0)
        mcount = dict(train=0, dev=0, test=0)
        mloss_ce = dict(train=0, dev=0, test=0)
        mloss_margin = dict(train=0, dev=0, test=0)
        macc_spk = dict(train=0, dev=0, test=0)
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
        ############################
        ### SORT FOR RNN PYTORCH ###
        ############################
        train_rr = [sort_reverse(x, train_len) for x in train_rr]
        dev_rr = [sort_reverse(x, dev_len) for x in dev_rr]
        test_rr = [sort_reverse(x, test_len) for x in test_rr]

        def save_model(model_name) :
            torch.save(ModelSerializer.convert_param_to_cpu(model.state_dict()), os.path.join(opts['result'], '{}.mdl'.format(model_name)))
            ModelSerializer.save_config(os.path.join(opts['result'], '{}.cfg'.format(model_name)), model.config)
            json.dump(opts, open(os.path.join(opts['result'], 'script.opts'), 'w'), indent=4)

        for set_name, set_rr, set_train_mode in [('train', train_rr, True), ('dev', dev_rr, False), ('test', test_rr, False)] :
            for rr in tqdm_wrapper(set_rr) :
                tic = timeit.default_timer()
                curr_feat_list = feat_iterator[set_name].get_feat_by_index(rr)
                feat_mat, feat_len = batch_speech(opts['gpu'], curr_feat_list, 
                        feat_sil=feat_sil, group=1, start_sil=1, end_sil=1)

                curr_key_list = feat_iterator[set_name].get_key_by_index(rr)
                curr_spk_list = [map_key2spk[x] for x in curr_key_list]

                _tmp_loss_ce, _tmp_acc = fn_batch_ce(feat_mat, feat_len, 
                        speaker_list=curr_spk_list, train_step=set_train_mode)
                _tmp_count = len(rr)
                # TODO : include margin loss later #
                assert_nan(_tmp_loss_ce)
                mloss_ce[set_name] += _tmp_loss_ce * _tmp_count
                mloss[set_name] += _tmp_loss_ce * _tmp_count
                macc_spk[set_name] += _tmp_acc * _tmp_count
                mcount[set_name] += _tmp_count
            pass

        info_header = ['set', 'loss', 'loss ce', 'loss margin', 'acc spk']
        info_table = []
        logger.info("Epoch %d -- lrate %f -- time %.2fs"%(ee, opts['lrate'], time.time() - start_time))
        for set_name in mloss.keys() :
            mloss[set_name] /= mcount[set_name]
            mloss_ce[set_name] /= mcount[set_name]
            mloss_margin[set_name] /= mcount[set_name]
            macc_spk[set_name] /= mcount[set_name]
            info_table.append([set_name, mloss[set_name], mloss_ce[set_name], 
                mloss_margin[set_name], macc_spk[set_name]])
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
        scheduler.step(mloss['dev'], ee)

    logger.info("best dev cost: %f at epoch %d"%(best_dev_loss, best_dev_loss_ep))
    pass
