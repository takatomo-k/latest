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
from torchev.utils.scheduler_util import ExponentialDecay
from torchev.optim.lr_scheduler import ReduceLROnPlateauEv

# utilbox #
from utilbox.data_util import iter_minibatches, iter_minibucket, iter_minibucket_block
from utilbox.math_util import assert_nan
from utilbox.parse_util import all_config_load
from utilbox.log_util import logger_stdout_file
from euterpe.data.data_generator import group_feat_timestep, feat_sil_from_stat, list_feat_sil_strip
from euterpe.data.data_iterator import TextIterator, DataIteratorNP
from euterpe.common.loader import DataLoader, LoaderDataType
from euterpe.common.batch_data import batch_speech_text
from euterpe.util import train_loop

from tensorboardX import SummaryWriter
from euterpe.model_tts.seq2seq.tacotron_core import TacotronType
import pandas as pd

DEBUG = False
def parse() :
    parser = argparse.ArgumentParser(description='training script for Tacotron Core')

    # format file V2 : model_cfg is defined by file
    parser.add_argument('--model_cfg', type=all_config_load, help='model configuration file')

    # data_cfg using new type npz loader #
    parser.add_argument('--data_cfg', type=str, default='config/dataset_wsj.json')

    # additional for bernoulli end #
    parser.add_argument('--coeff_bern', type=float, default=1)

    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--batchsize', type=int, default=20)
    parser.add_argument('--loss', type=str, choices=['L1', 'L2'], default=['L2'], nargs='+')
    parser.add_argument('--loss_freq_cfg', type=json.loads, default={'topn':0.25, 'coeff':1}, help='loss additional for raw spectrogram from 0-25 % freq range')
    parser.add_argument('--loss_diag_att_cfg', type=json.loads, default=None, 
            help='''loss penalty to encourage diagonal attention shape
            using exponential decay with params {"init_value":0.25, "decay_rate":0.94} ''')
    parser.add_argument('--mask_dec', action='store_true')
    parser.add_argument('--group', type=int, default=2, 
            help='group n-frame acoustic togather for 1 time step decoding (reduce number of step)')
    parser.add_argument('--pad_sil', type=int, default=2)

    parser.add_argument('--opt', type=str, default='Adam')
    parser.add_argument('--lrate', type=float, default=0.001) # default is not always converge
    parser.add_argument('--reducelr', type=json.loads, 
            default={'factor':0.5, 'patience':5, 'reset':False},
            help="""
            factor: decay lr * factor\n
            patience: wait x epoch until decay\n 
            reset: replace with new optimizer (only for Adam, Adagrad, etc)""") 
    parser.add_argument('--decay', type=float, default=1.0, help='decay lrate after no dev cost improvement')
    parser.add_argument('--grad_clip', type=float, default=5.0) # grad clip to prevent NaN
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='L2 weight decay regularization')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--result', type=str, default='expr_seq2seq_tacotron/dummy')
    parser.add_argument('--bucket', action='store_true')
    parser.add_argument('--block', type=int, default=-1)
    parser.add_argument('--sortagrad', type=int, default=-1)

    parser.add_argument('--cutoff', type=int, default=-1, help='cutoff frame larger than x')

    parser.add_argument('--update_part', type=str, default='all', help='freeze/unfreeze parameter')

    parser.add_argument('--model_pt', type=str, default=None, help='use pre-trained model for initialization')

    parser.add_argument('--save_interval', type=int, default=2, help='save model every x epoch')
    parser.add_argument('--mem', action='store_true', default=False)
    parser.add_argument('--strip_sil', action='store_true')

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
    data_cfg = yaml.load(open(opts['data_cfg']))

    feat_iterator = DataLoader.load_feat(data_cfg, 
            data_type=LoaderDataType.TYPE_NP, in_memory=opts['mem'])
    text_iterator = DataLoader.load_text(data_cfg)

    feat_stat = pickle.load(open(data_cfg['feat']['stat'], 'rb'))
    feat_sil = feat_sil_from_stat(feat_stat)

    print("Finish loading dataset ...") 
    
    NDIM = feat_iterator['train'].get_feat_dim() * group
    NVOCAB = len(text_iterator['train'].get_map_text2idx())

    if opts['model_pt'] is not None :
        model = ModelSerializer.load_config(os.path.join(opts['model_pt'], 'model.cfg')) 
        model.load_state_dict(torch.load(os.path.join(opts['model_pt'], 'model.mdl')))
        assert model.TYPE in [TacotronType.SINGLE_SPEAKER, TacotronType.MULTI_SPEAKER]
        print('[info] load pretrained model')
    else : 
        _model_cfg = opts['model_cfg']
        _model_cfg['enc_in_size'] = NVOCAB
        _model_cfg['dec_in_size'] = NDIM
        _model_cfg['dec_out_size'] = NDIM
        model = ModelSerializer.load_config(_model_cfg)
        assert model.TYPE in [TacotronType.SINGLE_SPEAKER, TacotronType.MULTI_SPEAKER]

    # send parameter to CPU/GPU
    if opts['gpu'] >= 0 :
        model.cuda(opts['gpu'])

    # load speaker vector if tacotron is MULTI_SPEAKER 
    if model.TYPE == TacotronType.MULTI_SPEAKER :
        feat_spkvec_iterator = DataIteratorNP(data_cfg['misc']['spkvec'])
        assert feat_spkvec_iterator.get_feat_dim() == model.speaker_emb_dim

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

    def criterion_diag_att(att_mat, dec_len, enc_len, std_dev=0.2, size_average=True) :
        if opts['loss_diag_att_cfg'] is None :
            return 0
        batch, max_dec_len, max_enc_len = att_mat.size() 
        loss = 0
        for bb in range(batch) :
            range_dec_len = torch.arange(0, dec_len[bb])
            range_enc_len = torch.arange(0, enc_len[bb])
            inv_normal_diag_mat = 1.0 - torch.exp(-((range_dec_len/dec_len[bb])[:, None] - (range_enc_len/enc_len[bb])[None, :])**2 / (2*std_dev**2))
            inv_normal_diag_mat = Variable(tensorauto(model, inv_normal_diag_mat)) # convert to device
            loss_bb = (att_mat[bb, 0:dec_len[bb], 0:enc_len[bb]] * inv_normal_diag_mat).sum()
            loss += loss_bb

        if size_average :
            loss /= batch

        return loss * scheduler_decay_diag_att.value

     
    # setting optimizer #
    list_opt_param = []
    if opts['update_part'] == 'all' :
        list_opt_param += list(model.parameters())
    else :
        raise NotImplementedError()

    opt = getattr(torch.optim, opts['opt'])(list_opt_param, lr=opts['lrate'], weight_decay=opts['weight_decay'])
    scheduler = ReduceLROnPlateauEv(opt, factor=opts['reducelr']['factor'], patience=opts['reducelr']['patience'], min_lr=opts['reducelr'].get('min_lr', 5e-5), verbose=True)
    scheduler_decay_diag_att = None
    if opts['loss_diag_att_cfg'] is not None :
        scheduler_decay_diag_att = ExponentialDecay(**opts['loss_diag_att_cfg'])

    def fn_batch(text_mat, text_len, feat_mat, feat_len, aux_info=None, train_step=True) :
        text_mat = Variable(text_mat)
        feat_mat_input = Variable(feat_mat[:, 0:-1])
        feat_mat_output = Variable(feat_mat[:, 1:])

        feat_mask = Variable(generate_seq_mask([x-1 for x in feat_len], opts['gpu']))

        feat_label_end = Variable(1. - generate_seq_mask([x-1-opts['pad_sil'] for x in feat_len], 
            opts['gpu'], max_len=feat_mask.size(1)))
        model.reset() 
        model.train(train_step)
        model.encode(text_mat, text_len)

        # additional input condition
        if model.TYPE == TacotronType.MULTI_SPEAKER :
            aux_info['speaker_vector'] = Variable(tensorauto(opts['gpu'], torch.from_numpy(np.stack(aux_info['speaker_vector']).astype('float32'))))
            model.set_aux_info(aux_info)

        batch, dec_len, _ = feat_mat_input.size()
        list_dec_core = []
        list_dec_core_bernoulli_end = []
        list_dec_att = []
        for ii in range(dec_len) :
            _dec_core_ii, _dec_att_ii, _dec_core_bernoulli_end = model.decode(feat_mat_input[:, ii], feat_mask[:, ii] if opts['mask_dec'] else None)
            list_dec_core.append(_dec_core_ii)
            list_dec_core_bernoulli_end.append(_dec_core_bernoulli_end)
            list_dec_att.append(_dec_att_ii['att_output']['p_ctx'])
            pass

        dec_core = torch.stack(list_dec_core, 1)
        dec_core_bernoulli_end = torch.cat(list_dec_core_bernoulli_end, 1)
        dec_att = torch.stack(list_dec_att, dim=1)

        # main : loss mel spectrogram #
        loss_core = criterion(dec_core, feat_mat_output, feat_mask)
        
        # optional : aux loss for lower frequency #
        loss_core_freq = 1 * criterion_freq(dec_core, feat_mat_output, feat_mask)

        loss_feat = loss_core + loss_core_freq

        # main : frame ending prediction #
        loss_core_bernoulli_end = F.binary_cross_entropy_with_logits(dec_core_bernoulli_end, feat_label_end) * opts['coeff_bern']
        acc_core_bernoulli_end = ((dec_core_bernoulli_end > 0.0) == (feat_label_end > 0.5)).float().mean()

        # optional : aux loss for encourage diagonal attention #
        loss_diag_att =  1 * criterion_diag_att(dec_att, dec_len=[x-1 for x in feat_len], enc_len=text_len)

        # combine all loss #
        loss = loss_feat + loss_core_bernoulli_end + loss_diag_att

        if train_step :
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), opts['grad_clip'])
            opt.step()

            # write report #
            tf_writer.add_scalar('loss', loss.data[0], global_step=tf_writer._n_iter)
            tf_writer.add_scalar('loss_feat', loss_feat.data[0], global_step=tf_writer._n_iter)
            tf_writer.add_scalar('loss_bern_end', loss_core_bernoulli_end.data[0], 
                    global_step=tf_writer._n_iter)
            if opts['loss_diag_att_cfg'] is not None :
                tf_writer.add_scalar('loss_diag_att', loss_diag_att.data[0], 
                        global_step=tf_writer._n_iter)
            tf_writer._n_iter += 1

        return loss.data.sum(), loss_feat.data.sum(), loss_core_bernoulli_end.data.sum(), acc_core_bernoulli_end.data.sum()
    
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

    ### FOR BI-RNN ENCODER ###
    train_text_len = text_iterator['train'].get_text_length()
    dev_text_len = text_iterator['dev'].get_text_length()
    test_text_len = text_iterator['test'].get_text_length()
    
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
    # save current script command #
    logger.info('{}'.format(' '.join([xi for xi in sys.argv])))

    # print model #
    logger.info('=== MODEL SUMMARY ===')
    logger.info(repr(model))
    logger.info('=====================')

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
    ee = 1
    while ee <= EPOCHS :
        start_time = time.time()
        
        mloss = dict(train=0, dev=0, test=0)
        mcount = dict(train=0, dev=0, test=0)
        mloss_feat = dict(train=0, dev=0, test=0)
        mloss_bernend = dict(train=0, dev=0, test=0)
        macc_bernend = dict(train=0, dev=0, test=0)
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
        ###############################################
        train_rr = [sort_reverse(x, train_text_len) for x in train_rr] 
        dev_rr = [sort_reverse(x, dev_text_len) for x in dev_rr] 
        test_rr = [sort_reverse(x, test_text_len) for x in test_rr] 
        ###############################################
        
        def save_model(model_name) :
            torch.save(ModelSerializer.convert_param_to_cpu(model.state_dict()), os.path.join(opts['result'], '{}.mdl'.format(model_name)))
            ModelSerializer.save_config(os.path.join(opts['result'], '{}.cfg'.format(model_name)), model.get_config())
            json.dump(opts, open(os.path.join(opts['result'], 'script.opts'), 'w'), indent=4)

        for set_name, set_rr, set_train_mode in [('train', train_rr, True), ('dev', dev_rr, False), ('test', test_rr, False)] :
            for rr in tqdm_wrapper(set_rr) :
                tic = timeit.default_timer()
                curr_key_list = feat_iterator[set_name].get_key_by_index(rr)
                curr_feat_list = feat_iterator[set_name].get_feat_by_key(curr_key_list)
                if opts['strip_sil'] :
                    curr_feat_list = list_feat_sil_strip(curr_feat_list)
                curr_label_list = text_iterator[set_name].get_text_by_key(curr_key_list)
                aux_info = None
                if model.TYPE == TacotronType.MULTI_SPEAKER :
                    curr_spkvec_list = feat_spkvec_iterator.get_feat_by_key(curr_key_list)
                    aux_info = {'speaker_vector':curr_spkvec_list}

                # print(1, timeit.default_timer() - tic); tic = timeit.default_timer()
                feat_mat, feat_len, text_mat, text_len = batch_speech_text(opts['gpu'], curr_feat_list, 
                        curr_label_list, feat_sil=feat_sil, group=opts['group'], start_sil=1, end_sil=opts['pad_sil'])
                # print(2, timeit.default_timer() - tic); tic = timeit.default_timer()
                _tmp_loss, _tmp_loss_feat, _tmp_loss_bernend, _tmp_acc_bernend = fn_batch(text_mat, text_len, 
                        feat_mat, feat_len, aux_info=aux_info, train_step=set_train_mode)
                # print(3, timeit.default_timer() - tic); tic = timeit.default_timer()
                _tmp_count = len(rr)
                assert_nan(_tmp_loss)
                mloss[set_name] += _tmp_loss * _tmp_count
                mloss_feat[set_name] += _tmp_loss_feat * _tmp_count
                mloss_bernend[set_name] += _tmp_loss_bernend * _tmp_count
                macc_bernend[set_name] += _tmp_acc_bernend * _tmp_count
                mcount[set_name] += _tmp_count
            pass

        info_header = ['set', 'loss', 'loss feat', 'loss bern end', 'acc bern end']
        info_table = []
        logger.info("Epoch %d -- lrate %f -- time %.2fs"%(ee, opt.param_groups[0]['lr'], time.time() - start_time))
        for set_name in mloss.keys() :
            mloss[set_name] /= mcount[set_name]
            mloss_feat[set_name] /= mcount[set_name]
            mloss_bernend[set_name] /= mcount[set_name]
            macc_bernend[set_name] /= mcount[set_name]
            info_table.append([set_name, mloss[set_name], mloss_feat[set_name], mloss_bernend[set_name], macc_bernend[set_name]])
        logger.info('\n'+tab.tabulate(info_table, headers=info_header, floatfmt='.3f', tablefmt='rst'))

        if (ee) % opts['save_interval'] == 0 :
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
        _sch_on_trigger = scheduler.step(mloss['dev'], ee)
        if _sch_on_trigger :
            if opts['reducelr'].get('reset', False) :
                opt = getattr(torch.optim, opts['opt'])(list_opt_param, lr=opt.param_groups[0]['lr'], weight_decay=opts['weight_decay'])
                logger.info('\n# scheduler triggered! reset opt & lr: {:.3e}'.format(opt.param_groups[0]['lr']))
        if scheduler_decay_diag_att is not None :
            scheduler_decay_diag_att.step(ee)

        # LAST STEP: increment epoch counter 
        ee += 1
         
        pass

    logger.info("final best dev loss %f at epoch %d"%(best_dev_loss, best_dev_loss_ep))
    pass
