import os
import sys

import pickle
import random
import json, yaml
import configargparse
import numpy as np
import itertools
import time
import operator
import tabulate as tab
from collections import Counter
import pandas as pd
from tensorboardX import SummaryWriter
from tqdm import tqdm

def tqdm_wrapper(obj) :
    return tqdm(obj, ascii=True, ncols=50)

# torch + torchev #
import torch
from torch import nn
from torch.nn import functional as F
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
from utilbox.print_util import printe

# euterpe #
from euterpe.common.loader import DataLoader, LoaderDataType
from euterpe.common.batch_data import batch_speech_text, batch_speech, \
        batch_text, batch_sorter, batch_select
from euterpe.config import constant
from euterpe.util import train_loop
from euterpe.data.data_generator import group_feat_timestep, feat_sil_from_stat
from euterpe.data.data_iterator import TextIterator, DataIteratorNP
from euterpe.model_tts.seq2seq.tacotron_core import TacotronType
from euterpe.common import generator_speech, generator_text
from euterpe.util import train_loop

def parse() :
    parser = configargparse.ArgParser(description='training script for speech chain between seq2seq ASR & Tacotron Core Multispeaker')
    ### UNIVERSAL param ###
    # main script config file #
    parser.add('-c', '--main_config', required=False, is_config_file=True, help='main script config file')

    # paired dataset #
    parser.add_argument('--data_pair_cfg', type=str, required=True)
    # unpaired dataset #
    parser.add_argument('--data_unpair_text_cfg', type=str, required=True)
    parser.add_argument('--data_unpair_speech_cfg', type=str, required=True)
    # others #
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--iter_per_epoch', type=int, default=None)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--result', type=str, required=True)

    parser.add_argument('--bucket', action='store_true')
    parser.add_argument('--block', type=int, default=-1)
    parser.add_argument('--sortagrad', type=int, default=-1)

    parser.add_argument('--save_interval', type=int, default=1, help='save model every x epoch')
    parser.add_argument('--mem', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)

    parser.add_argument('--coeff_pair', type=float, default=1)
    parser.add_argument('--coeff_unpair', type=float, default=1)

    ##################
    ### TTS params ###
    parser.add_argument('--tts_model_pt', type=str, default=None, help='use pre-trained TTS model for initialization', required=True)
    parser.add_argument('--tts_coeff_bern', type=float, default=1)

    parser.add_argument('--tts_batchsize', type=int, default=20)
    parser.add_argument('--tts_loss', type=str, choices=['L1', 'L2'], default=['L2'], nargs='+')
    parser.add_argument('--tts_loss_freq_cfg', type=yaml.load, default={'topn':0.25, 'coeff':1}, help='additional loss for raw spectrogram from certain freq range')
    parser.add_argument('--tts_loss_spk_emb', type=yaml.load, default=None, 
            help=
            '''
            additional loss for speaker embedding reconstruction
            {"type":["huber","L1","L2"], "coeff":1.0}
            ''')
    parser.add_argument('--tts_unpair_start_epoch', type=int, default=0)

    
    # TODO: remove loss_diag_att
    # parser.add_argument('--tts_loss_diag_att_cfg', type=yaml.load, default=None, 
            # help='''loss penalty to encourage diagonal attention shape
            # using exponential decay with params {"init_value":0.25, "decay_rate":0.94} ''')
    parser.add_argument('--tts_mask_dec', action='store_true', help='mask for TTS training')
    parser.add_argument('--tts_group', type=int, default=4, 
            help='group n-frame acoustic togather for 1 time step decoding (reduce number of step)')
    parser.add_argument('--tts_pad_sil', type=int, default=2)

    parser.add_argument('--tts_opt', type=str, default='Adam')
    parser.add_argument('--tts_lrate', type=float, default=0.001) # default is not always converge
    parser.add_argument('--tts_reducelr', type=yaml.load, 
            default={'factor':0.5, 'patience':3, 'reset':False},
            help="""
            factor: decay lr * factor\n
            patience: wait x epoch until decay\n 
            reset: replace with new optimizer (only for Adam, Adagrad, etc)""") 
    parser.add_argument('--tts_grad_clip', type=float, default=5.0) # grad clip to prevent NaN
    parser.add_argument('--tts_weight_decay', type=float, default=0, help='L2 weight decay regularization')
    parser.add_argument('--tts_cutoff', type=int, default=None, help='cutoff TTS speech frame larger than x')
    parser.add_argument('--tts_gen_cutoff', type=int, default=None, help='cutoff speech generated by TTS')
    parser.add_argument('--tts_spk_sample', type=str, default=None, help='sampling type')
    # parser.add_argument('--tts_gen_strip', action='store_true', default=False, help='auto pruning successful speech')

    # optional #
    parser.add_argument('--spkrec_model_pt', type=str, default=None, help='use pre-trained speaker recognition model for speaker embedding reconstruction loss')
    ##################
    ### ASR params ###
    parser.add_argument('--asr_model_pt', type=str, default=None, help='use pre-trained TTS model for initialization', required=True)
    parser.add_argument('--asr_batchsize', type=int, default=20)
    parser.add_argument('--asr_lbl_smooth', type=float, default=0.0, 
            help='label smoothing regularization')

    parser.add_argument('--asr_opt', type=str, default='Adam')
    parser.add_argument('--asr_lrate', type=float, default=0.001) # default is not always converge
    parser.add_argument('--asr_grad_clip', type=float, default=20.0) # grad clip to prevent NaN
    parser.add_argument('--asr_reducelr', type=yaml.load, 
            default={'factor':0.5, 'patience':3, 'reset':False},
            help="""
            factor: decay lr * factor\n
            patience: wait x epoch until decay\n 
            reset: replace with new optimizer (only for Adam, Adagrad, etc)""")
    parser.add_argument('--asr_cutoff', type=int, default=None, help='cutoff TTS speech frame larger than x')
    parser.add_argument('--asr_gen_search', type=yaml.load, default={'type':'greedy'}, 
            help='''unsupervised text generation method
            {"type":"greedy"}
            {"type":"beam", "kbeam":3, "chunk":5}
            ''')
    parser.add_argument('--asr_gen_cutoff', type=int, default=250, help='cutoff text generated by ASR')
    parser.add_argument('--asr_unpair_start_epoch', type=int, default=0)

    return parser.parse_args()

def assert_parse(opts) :
    assert opts['bucket'] == True
    assert opts['block'] == -1

if __name__ == '__main__':
    opts = vars(parse())
    assert_parse(opts)
    print(opts)

    # set default device #
    np.random.seed(opts['seed'])
    torch.manual_seed(opts['seed'])
    if opts['gpu'] >= 0 :
        torch.cuda.set_device(opts['gpu'])
        torch.cuda.manual_seed(opts['seed'])
    random.seed(opts['seed'])
    
    # dataset #
    data_pair_cfg = yaml.load(open(opts['data_pair_cfg']))
    feat_iterator = DataLoader.load_feat(data_pair_cfg, data_type=LoaderDataType.TYPE_NP, in_memory=opts['mem'])
    text_iterator = DataLoader.load_text(data_pair_cfg)
    map_text2idx = text_iterator['train'].get_map_text2idx()

    data_unpair_text_cfg = yaml.load(open(opts['data_unpair_text_cfg']))
    data_unpair_speech_cfg = yaml.load(open(opts['data_unpair_speech_cfg']))
    # feat_iterator['unpair'] = DataIteratorNP.load_feat(data_unpair_speech_cfg)['train']
    # text_iterator['unpair'] = DataIteratorNP.load_text(data_unpair_text_cfg)['train']
    feat_iterator['unpair'] = DataLoader.load_feat_single(data_unpair_speech_cfg['feat']['all'], 
            data_unpair_speech_cfg['feat']['set']['train'])
    text_iterator['unpair'] = DataLoader.load_text_single(data_unpair_speech_cfg['text']['all'], 
            data_unpair_speech_cfg['text']['set']['train'], vocab=map_text2idx)
    DataLoader._check_intersect(list(feat_iterator.values()))
    DataLoader._check_intersect(list(text_iterator.values()))
    assert text_iterator['train'].get_map_text2idx() == text_iterator['unpair'].get_map_text2idx()

    feat_stat = pickle.load(open(data_pair_cfg['feat']['stat'], 'rb'))
    feat_sil = feat_sil_from_stat(feat_stat)

    NVOCAB = len(text_iterator['train'].get_map_text2idx())
    NDIM_FEAT = feat_iterator['train'].get_feat_dim()

    printe("Finish loading dataset ...")

    # load model #
    model_asr = ModelSerializer.load_config_and_state(filename_state=opts['asr_model_pt'])
    model_tts = ModelSerializer.load_config_and_state(filename_state=opts['tts_model_pt'])

    if opts['tts_loss_spk_emb'] is not None :
        model_spkrec = ModelSerializer.load_config_and_state(filename_state=opts['spkrec_model_pt'])
    else :
        model_spkrec = None

    if opts['gpu'] >= 0 :
        model_asr.cuda(opts['gpu'])
        model_tts.cuda(opts['gpu'])
        if opts['tts_loss_spk_emb'] is not None :
            model_spkrec.cuda(opts['gpu'])

    # setting ASR criterion #  
    _class_weight = tensorauto(opts['gpu'], torch.ones(NVOCAB))
    _class_weight[constant.PAD] = 0
    _class_weight = Variable(_class_weight, requires_grad=False)
    asr_loss_ce = ElementwiseCrossEntropy(weight=_class_weight, 
            label_smoothing=opts['asr_lbl_smooth'])
    asr_opt = getattr(torch.optim, opts['asr_opt'])(model_asr.parameters(), lr=opts['asr_lrate'])
    asr_scheduler = ReduceLROnPlateauEv(asr_opt, factor=opts['asr_reducelr']['factor'], patience=opts['asr_reducelr']['patience'], min_lr=5e-5, verbose=True)

    # setting TTS criterion #
    # load speaker vector if tacotron is MULTI_SPEAKER 
    if model_tts.TYPE == TacotronType.MULTI_SPEAKER :
        feat_spkvec_iterator = DataIteratorNP(data_pair_cfg['misc']['spkvec'])
        assert feat_spkvec_iterator.get_feat_dim() == model_tts.speaker_emb_dim

    def tts_loss(input, target, mask, size_average=True, loss_type=opts['tts_loss'], mask_dec=opts['tts_mask_dec']) :
        loss = 0
        if 'L1' in loss_type :
            loss += torch.abs(input - target)
        if 'L2' in loss_type :
            loss += (input - target)**2 # batch x len x ndim #

        loss = torch.mean(loss, 2) # batch x len #
        if mask_dec == True :
            loss = loss * mask
        loss = torch.sum(loss) # sum all rest dim #
        if size_average :
            loss /= input.size(0)
        return loss

    def tts_loss_freq(input, target, mask, size_average=True, loss_cfg=opts['tts_loss_freq_cfg']) :
        """
        aux loss for prioritize optimizing loss on lower frequency
        """
        if loss_cfg is None :
            return Variable(torchauto(opts['gpu']).FloatTensor([0]))
        assert 0 < loss_cfg['topn'] <= 1
        assert loss_cfg['coeff'] > 0
        ndim = int(input.size(-1) * loss_cfg['topn'])
        loss = tts_loss(input[:, :, 0:ndim], target[:, :, 0:ndim], mask, size_average=True)
        return loss_cfg['coeff'] * loss

    def tts_loss_spk_emb(feat_mat, feat_len, target_emb, size_average=True, loss_cfg=opts['tts_loss_spk_emb']) :
        assert isinstance(feat_mat, Variable), "feat must be variable generate from TTS model"
        if loss_cfg is None :
            return Variable(torchauto(opts['gpu']).FloatTensor([0]))
        assert loss_cfg['type'] in ['huber', 'L1', 'L2', 'cosine']
        model_spkrec.reset()
        model_spkrec.eval() # set eval mode, no gradient update for model_spkrec #
        pred_emb = model_spkrec(feat_mat, feat_len)
        if loss_cfg['type'] == 'huber' :
            loss = nn.SmoothL1Loss(size_average=size_average)(pred_emb, target_emb)
        elif loss_cfg['type'] == 'L2' :
            loss = nn.MSELoss(size_average=size_average)(pred_emb, target_emb)
        elif loss_cfg['type'] == 'L1' :
            loss = nn.L1Loss(size_average=size_average)(pred_emb, target_emb)
        elif loss_cfg['type'] == 'cosine' :
            loss = (1-nn.CosineSimilarity()(pred_emb, target_emb)).sum() / (feat_mat.shape[0] if size_average else 1)
        else :
            raise NotImplementedError()
        return loss_cfg['coeff'] * loss
        pass

    tts_opt = getattr(torch.optim, opts['tts_opt'])(model_tts.parameters(), lr=opts['tts_lrate'])
    tts_scheduler = ReduceLROnPlateauEv(tts_opt, factor=opts['tts_reducelr']['factor'], 
            patience=opts['tts_reducelr']['patience'], min_lr=5e-5, verbose=True)
    
    def fn_batch_asr(model, feat_mat, feat_len, text_mat, text_len, train_step=True, coeff_loss=1) :
        # refit data #
        if max(feat_len) != feat_mat.shape[1] :
            feat_mat = feat_mat[:, 0:max(feat_len)]
        if max(text_len) != text_mat.shape[1] :
            text_mat = text_mat[:, 0:max(text_len)]

        if not isinstance(text_mat, Variable) :
            text_mat = Variable(text_mat) 
        if not isinstance(feat_mat, Variable) :
            feat_mat = Variable(feat_mat)
        text_input = text_mat[:, 0:-1]
        text_output = text_mat[:, 1:]
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
        loss = asr_loss_ce(pre_softmax.contiguous().view(batch * dec_len, -1), 
                text_output.contiguous().view(batch * dec_len)).view(batch, dec_len).sum(dim=1) / denominator
        loss = loss.mean() * coeff_loss

        acc = torch.max(pre_softmax, 2)[1].data.eq(text_output.data).masked_select(text_output.ne(constant.PAD).data).sum() / denominator.sum()
        if train_step == True :
            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), opts['asr_grad_clip'])
            asr_opt.step()
        return loss.data.sum(), acc.data.sum()

    def fn_batch_tts(model, text_mat, text_len, feat_mat, feat_len, aux_info=None, train_step=True, coeff_loss=1) :
        # refit data #
        if max(feat_len) != feat_mat.shape[1] :
            feat_mat = feat_mat[:, 0:max(feat_len)]
        if max(text_len) != text_mat.shape[1] :
            text_mat = text_mat[:, 0:max(text_len)]
        batch_size = text_mat.shape[0]
        if not isinstance(text_mat, Variable) :
            text_mat = Variable(text_mat) 
        if not isinstance(feat_mat, Variable) :
            feat_mat = Variable(feat_mat)
        feat_mat_input = feat_mat[:, 0:-1]
        feat_mat_output = feat_mat[:, 1:]

        feat_mask = Variable(generate_seq_mask([x-1 for x in feat_len], opts['gpu']))

        feat_label_end = Variable(1. - generate_seq_mask([x-1-opts['tts_pad_sil'] for x in feat_len], 
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
            _dec_core_ii, _dec_att_ii, _dec_core_bernoulli_end = model.decode(feat_mat_input[:, ii], feat_mask[:, ii] if opts['tts_mask_dec'] else None)
            list_dec_core.append(_dec_core_ii)
            list_dec_core_bernoulli_end.append(_dec_core_bernoulli_end)
            list_dec_att.append(_dec_att_ii['att_output']['p_ctx'])
            pass

        dec_core = torch.stack(list_dec_core, 1)
        dec_core_bernoulli_end = torch.cat(list_dec_core_bernoulli_end, 1)
        dec_att = torch.stack(list_dec_att, dim=1)

        # main : loss mel spectrogram #
        loss_core = tts_loss(dec_core, feat_mat_output, feat_mask)
        
        # optional : aux loss for lower frequency #
        loss_core_freq = 1 * tts_loss_freq(dec_core, feat_mat_output, feat_mask)

        loss_feat = loss_core + loss_core_freq

        # optional : aux loss for speaker embedding reconstruction #
        if model_tts.TYPE == TacotronType.MULTI_SPEAKER :
            loss_spk_emb = tts_loss_spk_emb(dec_core.view(batch_size, -1, NDIM_FEAT), 
                    [x*opts['tts_group'] for x in feat_len], aux_info['speaker_vector'])
        else :
            loss_spk_emb = Variable(torchauto(opts['gpu']).FloatTensor([0.0]))

        # main : frame ending prediction #
        loss_core_bernoulli_end = F.binary_cross_entropy_with_logits(dec_core_bernoulli_end, feat_label_end) * opts['tts_coeff_bern']
        acc_core_bernoulli_end = ((dec_core_bernoulli_end > 0.0) == (feat_label_end > 0.5)).float().mean()

        # combine all loss #
        loss = loss_feat + loss_core_bernoulli_end + loss_spk_emb
        loss = loss * coeff_loss

        # if train_step :
        if train_step == True :
            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), opts['tts_grad_clip'])
            tts_opt.step()

        return loss.data.sum(), loss_feat.data.sum(), loss_core_bernoulli_end.data.sum(), \
                loss_spk_emb.data.sum(), acc_core_bernoulli_end.data.sum()
   
    # PREPARE LOGGER + EXPR FOLDER

    # prepare model folder #
    RESULT_PATH = opts['result']
    RESULT_PATH_ASR = os.path.join(opts['result'], 'asr')
    RESULT_PATH_TTS = os.path.join(opts['result'], 'tts')
    if not os.path.exists(opts['result']) :
        os.makedirs(RESULT_PATH)
        os.makedirs(RESULT_PATH_ASR)
        os.makedirs(RESULT_PATH_TTS)
    else :
        if len(os.listdir(RESULT_PATH)) > 0:
            raise ValueError("Error : folder & data already existed !!!")

    def save_model(model, model_path, model_name) :
        torch.save(ModelSerializer.convert_param_to_cpu(model.state_dict()), os.path.join(model_path, '{}.mdl'.format(model_name)))
        ModelSerializer.save_config(os.path.join(model_path, '{}.cfg'.format(model_name)), model.get_config())
        json.dump(opts, open(os.path.join(opts['result'], 'script.opts'), 'w'), indent=2)

    logger = logger_stdout_file(os.path.join(RESULT_PATH, 'report.log'))
    # save current script command #
    logger.info('{}'.format(' '.join([xi for xi in sys.argv])))

    # CONSTANT #
    def sort_reverse(idx, length) : # for encoder mask must be sorted decreasing #
        return sorted(idx, key=lambda x : length[x], reverse=True)

    SET_NAMES = ['train', 'dev', 'test', 'unpair'] 
    
    # sort by length #
    feat_len = {}
    sorted_feat_idx = {}
    sorted_feat_len = {}
    text_len = {}
    sorted_text_idx = {}
    sorted_text_len = {}

    for set_name in SET_NAMES :
        feat_len[set_name] = feat_iterator[set_name].get_feat_length() 
        text_len[set_name] = text_iterator[set_name].get_text_length() 
        sorted_feat_idx[set_name] = np.argsort(feat_len[set_name]).tolist()
        sorted_text_idx[set_name] = np.argsort(text_len[set_name]).tolist()
        sorted_feat_len[set_name] = operator.itemgetter(*sorted_feat_idx[set_name])(feat_len[set_name])
        sorted_text_len[set_name] = operator.itemgetter(*sorted_text_idx[set_name])(text_len[set_name])

    tf_writer = SummaryWriter(log_dir=os.path.join(RESULT_PATH, 'tfboard.event'))
    tf_writer.step_counter = Counter()
    def auto_writer_scalar(tag, scalar_value) :
        step = tf_writer.step_counter[tag]
        tf_writer.add_scalar(tag, scalar_value, global_step=step)
        tf_writer.step_counter[tag] += 1

    exc_idx = {}
    for set_name, limit, set_len in [
            ('train_feat', opts['asr_cutoff'], feat_len['train']),
            ('dev_feat', opts['asr_cutoff'], feat_len['dev']),
            ('test_feat', opts['asr_cutoff'], feat_len['test']),
            ('unpair_feat', opts['asr_cutoff'], feat_len['unpair']),
            ('train_text', opts['tts_cutoff'], feat_len['train']),
            ('dev_text', opts['tts_cutoff'], feat_len['dev']),
            ('test_text', opts['tts_cutoff'], feat_len['test']),
            ('unpair_text', opts['tts_cutoff'], feat_len['unpair']),
            ] :
        exc_idx[set_name] = set(map(lambda x:x[0], filter(lambda x:x[1]>limit or limit is None, enumerate(set_len))))
        logger.info('[info] exclude set {} : {} utts from total {}'.format(set_name, len(exc_idx[set_name]), len(set_len)))

    # TRAIN & EVAL LOOP #
    EPOCHS = opts['epoch'] 
    FEAT_BATCHSIZE = opts['asr_batchsize'] 
    TEXT_BATCHSIZE = opts['tts_batchsize'] 
    ITER_PER_EPOCH = opts['iter_per_epoch']

    ee = 1

    # logging #
    HEADER_ASR = ['SET', '[ASR] LOSS CE', '[ASR] ACC']
    HEADER_TTS = ['SET', '[TTS] LOSS', '[TTS] LOSS FEAT', '[TTS] LOSS BCE', '[TTS] LOSS SPK EMB', '[TTS] ACC']

    set_name = ['train', 'dev', 'test', 'unpair']
    feat_rr = dict((x, None) for x in set_name)
    text_rr = dict((x, None) for x in set_name)
    for set_name in ['dev', 'test'] :
        feat_rr[set_name] = list(iter_minibucket(sorted_feat_idx[set_name],
            FEAT_BATCHSIZE, shuffle=False, 
            excludes=exc_idx[set_name+'_feat']))
        text_rr[set_name] = list(iter_minibucket(sorted_text_idx[set_name],
            TEXT_BATCHSIZE, shuffle=False, 
            excludes=exc_idx[set_name+'_text']))

    best_asr_dev = {'loss':2**64, 'ep':0} # loss & epoch #
    best_tts_dev = {'loss':2**64, 'ep':0} # loss & epoch #
    logger.info("=====START=====")
    while ee <= EPOCHS :
        start_time = time.time()

        m_asr_loss = dict((x, 0) for x in SET_NAMES)
        m_asr_acc = dict((x, 0) for x in SET_NAMES)
        m_asr_count = dict((x, 0) for x in SET_NAMES)
        m_asr_iter_idx = dict((x, 0) for x in SET_NAMES) 
        m_asr_gen_info = {'valid':0, 'total':0}

        m_tts_loss = dict((x, 0) for x in SET_NAMES)
        m_tts_loss_feat = dict((x, 0) for x in SET_NAMES)
        m_tts_loss_bce = dict((x, 0) for x in SET_NAMES)
        m_tts_loss_spk_emb = dict((x, 0) for x in SET_NAMES)
        m_tts_acc = dict((x, 0) for x in SET_NAMES)
        m_tts_count = dict((x, 0) for x in SET_NAMES)
        m_tts_iter_idx = dict((x, 0) for x in SET_NAMES)
        m_tts_gen_info = {'valid':0, 'total':0}

        ### train/dev/test shortcut ###
        def auto_writer_info_asr(set_name, loss, acc) :
            auto_writer_scalar('ASR/{}_loss'.format(set_name), loss)
            auto_writer_scalar('ASR/{}_acc'.format(set_name), acc)

        def auto_writer_info_tts(set_name, loss, loss_feat, loss_bce, loss_spk_emb, acc) :
            auto_writer_scalar('TTS/{}_loss'.format(set_name), loss)
            auto_writer_scalar('TTS/{}_loss_feat'.format(set_name), loss_feat)
            auto_writer_scalar('TTS/{}_loss_bce'.format(set_name), loss_bce)
            auto_writer_scalar('TTS/{}_loss_spk_emb'.format(set_name), loss_spk_emb)

            auto_writer_scalar('TTS/{}_acc'.format(set_name), acc)

        def iter_asr(set_name, set_train_mode, set_rr) :
            assert set_name in ['train', 'dev', 'test']
            rr = set_rr
            rr = sort_reverse(rr, feat_len[set_name])
            rr_key = feat_iterator[set_name].get_key_by_index(rr)
            curr_feat_list = feat_iterator[set_name].get_feat_by_key(rr_key)
            curr_text_list = text_iterator[set_name].get_text_by_key(rr_key)

            curr_feat_mat, curr_feat_len, curr_text_mat, curr_text_len = batch_speech_text(opts['gpu'], curr_feat_list, curr_text_list) 

            _loss, _acc = fn_batch_asr(model_asr, curr_feat_mat, curr_feat_len, curr_text_mat, curr_text_len, 
                    train_step=set_train_mode, coeff_loss=opts['coeff_pair'])
            _loss /= opts['coeff_pair']
            assert_nan(_loss)
            _count = len(rr)
            m_asr_loss[set_name] += _loss * _count
            m_asr_acc[set_name] += _acc * _count
            m_asr_count[set_name] += _count
            if tf_writer is not None :
                auto_writer_info_asr(set_name, _loss, _acc)
       
        def iter_tts(set_name, set_train_mode, set_rr) :
            assert set_name in ['train', 'dev', 'test']
            rr = set_rr
            rr = sort_reverse(rr, text_len[set_name])
            rr_key = text_iterator[set_name].get_key_by_index(rr)
            curr_feat_list = feat_iterator[set_name].get_feat_by_key(rr_key)
            curr_text_list = text_iterator[set_name].get_text_by_key(rr_key)
            if model_tts.TYPE == TacotronType.MULTI_SPEAKER :
                curr_spkvec_list = feat_spkvec_iterator.get_feat_by_key(rr_key)
                curr_aux_info = {'speaker_vector':curr_spkvec_list}
            else :
                curr_aux_info = None
            curr_feat_mat, curr_feat_len, curr_text_mat, curr_text_len = batch_speech_text(opts['gpu'], curr_feat_list, 
                    curr_text_list, feat_sil=feat_sil, group=opts['tts_group'], start_sil=1, end_sil=opts['tts_pad_sil'])
            _loss, _loss_feat, _loss_bce_fend, _loss_spk_emb, _acc_fend = fn_batch_tts(model_tts, curr_text_mat, curr_text_len, 
                    curr_feat_mat, curr_feat_len, aux_info=curr_aux_info, train_step=set_train_mode, coeff_loss=opts['coeff_pair'])
            _loss /= opts['coeff_pair']
            assert_nan(_loss)
            _count = len(rr)
            m_tts_loss[set_name] += _loss * _count
            m_tts_loss_feat[set_name] += _loss_feat * _count
            m_tts_loss_bce[set_name] += _loss_bce_fend * _count
            m_tts_loss_spk_emb[set_name] += _loss_spk_emb * _count
            m_tts_acc[set_name] += _acc_fend * _count
            m_tts_count[set_name] += _count

            if tf_writer is not None :
                auto_writer_info_tts(set_name, _loss, _loss_feat, _loss_bce_fend, _loss_spk_emb, _acc_fend)

        def iter_cycle_tts2asr(set_name, set_train_mode, set_rr) :
            rr = set_rr
            rr = sort_reverse(rr, text_len[set_name])
            rr_key = text_iterator[set_name].get_key_by_index(rr)
            curr_text_list = text_iterator[set_name].get_text_by_key(rr_key)
            if model_tts.TYPE == TacotronType.MULTI_SPEAKER :
                if opts['tts_spk_sample'] is None :
                    curr_spkvec_list = feat_spkvec_iterator.get_feat_by_key(rr_key)
                elif opts['tts_spk_sample'] == 'uniform' : 
                    _sample_rr_key = random.sample(feat_iterator[set_name].key, k=len(set_rr))
                    curr_spkvec_list = feat_spkvec_iterator.get_feat_by_key(_sample_rr_key)
                else :
                    raise NotImplementedError()
                curr_aux_info = {'speaker_vector':curr_spkvec_list}
            else :
                curr_aux_info = None
            curr_text_mat, curr_text_len = batch_text(opts['gpu'], curr_text_list)

            curr_pred_feat_list, curr_pred_feat_len, curr_pred_att_mat = generator_speech.decode_greedy_pred(model_tts, curr_text_mat, curr_text_len, 
                    group=opts['tts_group'], feat_sil=feat_sil, aux_info=curr_aux_info, 
                    max_target=opts['tts_gen_cutoff']//opts['tts_group'])

            # filter bad speech #
            curr_pred_quality = generator_text.eval_gen_text_quality(None, curr_pred_feat_len, None)
            curr_pred_valid_idx = [x for x, y in enumerate(curr_pred_quality) if y == 1]

            m_tts_gen_info['total'] += len(rr)
            m_tts_gen_info['valid'] += len(curr_pred_valid_idx)
            if len(curr_pred_valid_idx) == 0 :
                return None

            curr_pred_feat_list = batch_select(curr_pred_feat_list, curr_pred_valid_idx)
            curr_pred_feat_len = batch_select(curr_pred_feat_len, curr_pred_valid_idx)
            
            curr_text_mat = batch_select(curr_text_mat, curr_pred_valid_idx) 
            curr_text_len = batch_select(curr_text_len, curr_pred_valid_idx)

            # zip & sort dec #
            curr_pred_feat_list = batch_sorter(curr_pred_feat_list, curr_pred_feat_len)
            curr_text_mat = batch_sorter(curr_text_mat, curr_pred_feat_len)
            curr_text_len = batch_sorter(curr_text_len, curr_pred_feat_len)
            curr_pred_feat_len = batch_sorter(curr_pred_feat_len, curr_pred_feat_len) # sort key must be on the last step

            
            curr_pred_feat_mat, curr_pred_feat_len = batch_speech(opts['gpu'], curr_pred_feat_list)
            # if sorted(curr_pred_feat_len, reverse=True) != curr_pred_feat_len :
                # import ipdb; ipdb.set_trace()
            _loss, _acc = fn_batch_asr(model_asr, curr_pred_feat_mat, curr_pred_feat_len, 
                    curr_text_mat, curr_text_len, 
                    train_step=set_train_mode, coeff_loss=opts['coeff_unpair'])
            _loss /= opts['coeff_unpair']
            assert_nan(_loss)
            _count = len(rr)
            m_asr_loss[set_name] += _loss * _count
            m_asr_acc[set_name] += _acc * _count
            m_asr_count[set_name] += _count
            if tf_writer is not None :
                auto_writer_info_asr(set_name, _loss, _acc)

        def iter_cycle_asr2tts(set_name, set_train_mode, set_rr) :
            rr = set_rr
            rr = sort_reverse(rr, feat_len[set_name])
            rr_key = feat_iterator[set_name].get_key_by_index(rr)
            curr_feat_list = feat_iterator[set_name].get_feat_by_key(rr_key)
            curr_feat_mat, curr_feat_len = batch_speech(opts['gpu'], curr_feat_list, 
                    feat_sil=feat_sil, group=opts['tts_group'], start_sil=1, end_sil=opts['tts_pad_sil'])
            # modified feature for ASR #
            curr_feat_mat_for_asr = curr_feat_mat[:, 1:-opts['tts_pad_sil']].contiguous().view(len(set_rr), -1, NDIM_FEAT)
            curr_feat_len_for_asr = [len(x) for x in curr_feat_list]

            if opts['asr_gen_search']['type'] == 'greedy' :
                curr_pred_text_list, curr_pred_text_len, curr_pred_att_mat = generator_text.greedy_search(
                        model_asr, curr_feat_mat_for_asr, curr_feat_len_for_asr, 
                        map_text2idx=map_text2idx, max_target=opts['asr_gen_cutoff'])
            elif opts['asr_gen_search']['type'] == 'beam' :
                curr_pred_text_list, curr_pred_text_len = [], []
                for ii in range(0, len(rr), opts['asr_gen_search']['chunk']) :
                    _start_ii = ii
                    _end_ii = min(ii+opts['asr_gen_search']['chunk'], len(rr))
                    curr_pred_text_list_ii, curr_pred_text_len_ii, _ = generator_text.beam_search(
                            model_asr, curr_feat_mat_for_asr[_start_ii:_end_ii], curr_feat_len_for_asr[_start_ii:_end_ii], 
                            map_text2idx=map_text2idx, max_target=opts['asr_gen_cutoff'], 
                            kbeam=opts['asr_gen_search']['kbeam'])
                    curr_pred_text_list.extend(curr_pred_text_list_ii)
                    curr_pred_text_len.extend(curr_pred_text_len_ii)
            if model_tts.TYPE == TacotronType.MULTI_SPEAKER :
                curr_spkvec_list = feat_spkvec_iterator.get_feat_by_key(rr_key)

            # TODO: filter bad text #
            curr_pred_quality = generator_text.eval_gen_text_quality(None, curr_pred_text_len, None)
            curr_pred_valid_idx = [x for x,y in enumerate(curr_pred_quality) if y == 1]

            m_asr_gen_info['total'] += len(rr)
            m_asr_gen_info['valid'] += len(curr_pred_valid_idx)
            if len(curr_pred_valid_idx) == 0 :
                return None

            curr_pred_text_list = batch_select(curr_pred_text_list, curr_pred_valid_idx)
            if model_tts.TYPE == TacotronType.MULTI_SPEAKER :
                curr_spkvec_list = batch_select(curr_spkvec_list, curr_pred_valid_idx)
            curr_pred_text_len = batch_select(curr_pred_text_len, curr_pred_valid_idx)

            curr_feat_mat = batch_select(curr_feat_mat, curr_pred_valid_idx)
            curr_feat_len = batch_select(curr_feat_len, curr_pred_valid_idx)

            # zip & sort dec #
            curr_pred_text_list = batch_sorter(curr_pred_text_list, curr_pred_text_len)
            if model_tts.TYPE == TacotronType.MULTI_SPEAKER :
                curr_spkvec_list = batch_sorter(curr_spkvec_list, curr_pred_text_len)
            curr_feat_mat = batch_sorter(curr_feat_mat, curr_pred_text_len)
            curr_feat_len = batch_sorter(curr_feat_len, curr_pred_text_len)
            curr_pred_text_len = batch_sorter(curr_pred_text_len, curr_pred_text_len) # sort key must be on the last step


            curr_pred_text_mat, curr_pred_text_len = batch_text(opts['gpu'], curr_pred_text_list, )

            if model_tts.TYPE == TacotronType.MULTI_SPEAKER :
                curr_aux_info = {'speaker_vector':curr_spkvec_list}
            else :
                curr_aux_info = None
            _loss, _loss_feat, _loss_bce_fend, _loss_spk_emb, _acc_fend = fn_batch_tts(model_tts, 
                    curr_pred_text_mat, curr_pred_text_len, 
                    curr_feat_mat, curr_feat_len, aux_info=curr_aux_info, 
                    train_step=set_train_mode, coeff_loss=opts['coeff_unpair'])
            _loss /= opts['coeff_unpair']

            assert_nan(_loss)
            _count = len(curr_pred_text_list)
            m_tts_loss[set_name] += _loss * _count
            m_tts_loss_feat[set_name] += _loss_feat * _count
            m_tts_loss_bce[set_name] += _loss_bce_fend * _count
            m_tts_loss_spk_emb[set_name] += _loss_spk_emb * _count
            m_tts_acc[set_name] += _acc_fend * _count
            m_tts_count[set_name] += _count
    
            if tf_writer is not None :
                auto_writer_info_tts(set_name, _loss, _loss_feat, _loss_bce_fend, _loss_spk_emb, _acc_fend)

        # ============================== #

        for iter_ee in tqdm_wrapper(range(0, ITER_PER_EPOCH)) :
            # check if idx is None or idx == len(idx) #
            for set_name in ['train','unpair'] :
                if feat_rr[set_name] is None or m_asr_iter_idx[set_name] >= len(feat_rr[set_name]) :
                    feat_rr[set_name] = list(iter_minibucket(sorted_feat_idx[set_name], FEAT_BATCHSIZE, shuffle=(ee > opts['sortagrad'])))
                    m_asr_iter_idx[set_name] = 0
                if text_rr[set_name] is None or m_tts_iter_idx[set_name] >= len(text_rr[set_name]) :
                    text_rr[set_name] = list(iter_minibucket(sorted_text_idx[set_name], TEXT_BATCHSIZE, shuffle=(ee > opts['sortagrad'])))
                    m_tts_iter_idx[set_name] = 0
             
            # train pair #
            # ASR #
            iter_asr('train', True, feat_rr['train'][m_asr_iter_idx['train']])
            # TTS #
            iter_tts('train', True, text_rr['train'][m_tts_iter_idx['train']])

            # train unpair #
            # speech -> ASR -> text -> TTS -> speech #
            if ee >= opts['tts_unpair_start_epoch'] :
                iter_cycle_asr2tts('unpair', True, feat_rr['unpair'][m_asr_iter_idx['unpair']])

            # text -> TTS -> speech -> ASR -> text #
            if ee >= opts['asr_unpair_start_epoch'] :
                iter_cycle_tts2asr('unpair', True, text_rr['unpair'][m_tts_iter_idx['unpair']])

            # increment all #
            m_asr_iter_idx['train'] += 1
            m_asr_iter_idx['unpair'] += 1
            m_tts_iter_idx['train'] += 1
            m_tts_iter_idx['unpair'] += 1
            if opts['debug'] :
                break
            pass

        # dev & test #
        for set_name, set_train_mode in [('dev', False), ('test', False)] :
            # ASR #
            for rr in tqdm_wrapper(list(feat_rr[set_name])) :
                iter_asr(set_name, set_train_mode=set_train_mode, set_rr=rr)
                if opts['debug'] :
                    break
            # TTS #
            for rr in tqdm_wrapper(list(text_rr[set_name])) :
                iter_tts(set_name, set_train_mode=set_train_mode, set_rr=rr)
                if opts['debug'] :
                    break
        
        # print report #
        logger.info("\nEpoch: {} - lrate ASR {:.3e} - lrate TTS {:.3e} - time {:.2f}s".format(ee, asr_opt.param_groups[0]['lr'], tts_opt.param_groups[0]['lr'], time.time() - start_time))
        info_table_tts = []
        info_table_asr = []
        for set_name in SET_NAMES :
            m_asr_loss[set_name] /= max(m_asr_count[set_name], 1)
            m_asr_acc[set_name] /= max(m_asr_count[set_name], 1)
            m_tts_loss[set_name] /= max(m_tts_count[set_name], 1)
            m_tts_loss_feat[set_name] /= max(m_tts_count[set_name], 1)
            m_tts_loss_bce[set_name] /= max(m_tts_count[set_name], 1)
            m_tts_acc[set_name] /= max(m_tts_count[set_name], 1)
            m_tts_loss_spk_emb[set_name] /= max(m_tts_count[set_name], 1)

            info_table_asr.append([set_name, m_asr_loss[set_name], m_asr_acc[set_name]])
            info_table_tts.append([set_name, m_tts_loss[set_name], m_tts_loss_feat[set_name], m_tts_loss_bce[set_name], m_tts_loss_spk_emb[set_name], m_tts_acc[set_name]])
        logger.info('\n'+tab.tabulate(info_table_asr, headers=HEADER_ASR, floatfmt='.3f', tablefmt='rst'))
        logger.info('\n'+tab.tabulate(info_table_tts, headers=HEADER_TTS, floatfmt='.3f', tablefmt='rst'))
        logger.info('\n[info] valid gen speech: {}/{}, text: {}/{}'.format(m_tts_gen_info['valid'], m_tts_gen_info['total'], m_asr_gen_info['valid'], m_asr_gen_info['total']))

        # save model #
        if (ee) % opts['save_interval'] == 0 :
            save_model(model_asr, RESULT_PATH_ASR, 'model_{}'.format(ee))
            save_model(model_tts, RESULT_PATH_TTS, 'model_{}'.format(ee))

        # save best model #
        if m_tts_loss['dev'] < best_tts_dev['loss'] :
            best_tts_dev['loss'] = m_tts_loss['dev']
            best_tts_dev['ep'] = ee
            save_model(model_tts, RESULT_PATH_TTS, 'best_model'.format(ee))
            logger.info("[info] achieved best dev TTS loss ! serialized the model")
        if m_asr_loss['dev'] < best_asr_dev['loss'] :
            best_asr_dev['loss'] = m_asr_loss['dev']
            best_asr_dev['ep'] = ee
            save_model(model_asr, RESULT_PATH_ASR, 'best_model'.format(ee))
            logger.info("[info] achieved best dev ASR loss ! serialized the model")

        # update scheduler
        _sch_on_trigger = asr_scheduler.step(m_asr_loss['dev'], ee)
        if _sch_on_trigger :
            if opts['asr_reducelr'].get('reset', False) :
                asr_opt = getattr(torch.optim, opts['asr_opt'])(model_asr.parameters(), lr=asr_opt.param_groups[0]['lr'])
                asr_scheduler.optimizer = asr_opt
                logger.info('\t# ASR scheduler triggered! reset opt & reduce lr: {:.3e}'.format(asr_opt.param_groups[0]['lr']))

        _sch_on_trigger = tts_scheduler.step(m_asr_loss['dev'], ee)
        if _sch_on_trigger :
            if opts['tts_reducelr'].get('reset', False) :
                tts_opt = getattr(torch.optim, opts['tts_opt'])(model_tts.parameters(), lr=tts_opt.param_groups[0]['lr'])
                tts_scheduler.optimizer = tts_opt
                logger.info('\t# TTS scheduler triggered! reset opt & reduce lr: {:.3e}'.format(tts_opt.param_groups[0]['lr']))
        # increment epoch
        ee += 1
        pass
    logger.info("final best dev ASR {:.4f} at epoch {}".format(best_asr_dev['loss'], best_asr_dev['ep']))
    logger.info("final best dev TTS {:.4f} at epoch {}".format(best_tts_dev['loss'], best_tts_dev['ep']))
