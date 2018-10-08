import os
import sys
from __init__ import *

import argparse
import json
import numpy as np
import itertools
import time
import timeit
import operator
import pickle
import tabulate as tab
from datetime import datetime
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
from torchev.utils.train_util import LinearScheduler

# utilbox #
from utilbox.data_util import iter_minibatches, iter_minibucket, iter_minibucket_block
from utilbox.math_util import assert_nan
from utilbox.log_util import logger_stdout_file
from tensorboard import SummaryWriter
from data.data_generator import group_feat_timestep, feat_sil_from_stat
from data.data_iterator import DataIterator, TextIterator
from model_vc.seq2seq.static_model import GeneratorStaticRNN
from model_vc.seq2seq.dynamic_model import GeneratorSeq2SeqRNN
from model_vc.seq2seq.disc import DiscriminatorCNN
from common.loader import loader_seq2seq_single_feat
from common.batch_data import batch_speech
import pandas as pd

DEBUG = False
def parse() :
    parser = argparse.ArgumentParser(description='voice conversion cyclegan (WORLD)')
   
    # TODO : change architecture from args #

    parser.add_argument('--data_cfg', type=str, default='config/dataset_wsj.json')

    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--batchsize', type=int, default=20)
    parser.add_argument('--loss', type=str, choices=['l1', 'l2'], default='l2')
    parser.add_argument('--mask_dec', action='store_true')
    parser.add_argument('--group', type=int, default=2, 
            help='group n-frame acoustic togather for 1 time step decoding (reduce number of step)')
    parser.add_argument('--pad_sil', type=int, default=2)

    parser.add_argument('--opt', type=str, default='Adam')
    parser.add_argument('--lrate', type=float, default=0.001) # default is not always converge
    parser.add_argument('--decay', type=float, default=1.0, help='decay lrate after no dev cost improvement')
    parser.add_argument('--grad_clip', type=float, default=20.0) # grad clip to prevent NaN
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--result', type=str, default='expr_dummy/dummy')
    parser.add_argument('--bucket', action='store_true')
    parser.add_argument('--block', type=int, default=-1)
    parser.add_argument('--sortagrad', type=int, default=-1)

    parser.add_argument('--cutoff', type=int, default=-1, help='cutoff frame larger than x')
    parser.add_argument('--spk_a', type=str, required=True)
    parser.add_argument('--spk_b', type=str, required=True)

    parser.add_argument('--mtype', type=str, choices=['static', 'dyn'], default='dyn')

    parser.add_argument('--c_pair', type=float, required=True, help='paired loss coeff')
    parser.add_argument('--c_gan', type=float, required=True, help='gan coeff')
    parser.add_argument('--c_cycle', type=float, required=True, help='cycle consistency coeff')
    parser.add_argument('--c_idt', type=float, required=True, help='identity coeff')
    parser.add_argument('--gan_type', type=str, choices=['none', 'wgan', 'lsgan'], 
            default='lsgan', help='none -> not using gan')
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
    data_cfg = json.load(open(opts['data_cfg']))
    feat_iterator = loader_seq2seq_single_feat(data_cfg)
    feat_stat = pickle.load(open(data_cfg['feat_stat'], 'rb'))
    feat_sil = feat_sil_from_stat(feat_stat)
    
    # load key for speaker a & b #
    data_info = pd.read_csv(data_cfg['info'], sep='\t')
    list_key_spk_a = data_info[data_info.SPK == opts['spk_a']].KEY.tolist()
    list_key_spk_b = data_info[data_info.SPK == opts['spk_b']].KEY.tolist()
    
    # load key for paired utt between spk a & b #
    pair_info = None
    if 'pair_info' in data_cfg :
        pair_info = pd.read_csv(data_cfg['pair_info'], sep='\t')
        pair_info = dict([(x, y) for (x, y) in zip(pair_info.SPKA.tolist(), pair_info.SPKB.tolist())])
    
    print("Finish loading dataset ...") 
    
    NDIM = feat_iterator.get_feat_dim() * group
    NDIM_SECOND = NDIM # dummy, we don't need this # 

    if opts['mtype'] == 'static' :
        # create generative model A -> B #
        model_gen_a2b = GeneratorStaticRNN(NDIM, NDIM) # TODO : configure parameter #
        # create generative model B -> A #
        model_gen_b2a = GeneratorStaticRNN(NDIM, NDIM) # TODO : configure parameter #
    else :
        model_gen_a2b = GeneratorSeq2SeqRNN(NDIM, NDIM, NDIM)
        model_gen_b2a = GeneratorSeq2SeqRNN(NDIM, NDIM, NDIM)

    # create discriminator domain A #
    model_disc_a = DiscriminatorCNN(NDIM)
    # create discriminator domain B #
    model_disc_b = DiscriminatorCNN(NDIM)

    def criterion_recon(input, target, mask, size_average=True) :
        # l1 or l2 # 
        if opts['loss'] == 'l1' :
            loss = torch.abs(input - target)
        elif opts['loss'] == 'l2' :
            loss = (input - target)**2 # batch x len x ndim #
        loss = torch.mean(loss, 2) # batch x len #
        # TODO masking #
        # if opts['mask_dec'] :
            # loss = loss * mask
        loss = torch.mean(loss) # sum all rest dim #
        if size_average :
            loss /= input.size(0)
        return loss
        pass
    # TODO discriminator criterion (WGAN ? GAN ?)
    def criterion_gan(input, target, mask, size_average=True) :
        # convert to 2d #
        batch, seq_len = input.size()
        input_1d = input.contiguous().view(-1)
        # TODO : use mask ? #
        if opts['gan_type'] == 'none' :
            return None
        elif opts['gan_type'] == 'gan' :
            # normal gan loss #
            target_1d = Variable(torchauto(opts['gpu']).FloatTensor(input_1d.size()).fill_(target))
            return F.binary_cross_entropy_with_logits(input_1d, target_1d)
        elif opts['gan_type'] == 'wgan' :
            return torch.mean(input_1d * (-1 if target else 1))
        elif opts['gan_type'] == 'lsgan' :
            target_1d = Variable(torchauto(opts['gpu']).FloatTensor(input_1d.size()).fill_(target))
            return torch.mean((input_1d - target_1d)**2)
        else :
            raise NotImplementedError
        pass

    if opts['gpu'] >= 0 :
        model_gen_a2b.cuda(opts['gpu'])
        model_gen_b2a.cuda(opts['gpu'])
        model_disc_a.cuda(opts['gpu'])
        model_disc_b.cuda(opts['gpu'])
        pass

    # setting optimizer #
    optim_gen_a2b = getattr(torch.optim, opts['opt'])(model_gen_a2b.parameters(), lr=opts['lrate'])
    optim_gen_b2a = getattr(torch.optim, opts['opt'])(model_gen_b2a.parameters(), lr=opts['lrate'])
    optim_disc_a = getattr(torch.optim, opts['opt'])(model_disc_a.parameters(), lr=opts['lrate'])
    optim_disc_b = getattr(torch.optim, opts['opt'])(model_disc_b.parameters(), lr=opts['lrate'])
    
    # setting scheduler #
    scheduler_coeff_gan = LinearScheduler(init=0, delta=0.1, start_epoch=5, max_val=1.0)
    

    def fn_generate_static(model_gen, feat_source, feat_len, train_step=True) :
        if not isinstance(feat_source, Variable) :
            feat_source = Variable(feat_source)

        model_gen.train(train_step)
        res = model_gen.transcode(feat_source, feat_len)
        return res

    def fn_generate_dynamic(model_gen, feat_source, feat_source_len, train_step=True) :
        if not isinstance(feat_source, Variable) :
            feat_source = Variable(feat_source)
        # TODO : how to stop ?! #
        pass

    # supervised generation only for dynamic model (by using autoregressive) #
    def fn_generate_dynamic_supervised(model_gen, feat_source, feat_source_len, 
            feat_target, feat_target_len, train_step=True, 
            tfboard_writer=None, niter=0, opt=None, model_name='') :
        if not isinstance(feat_source, Variable) :
            feat_source = Variable(feat_source)

        if not isinstance(feat_target, Variable) :
            feat_target = Variable(feat_target)

        feat_target_input = feat_target[:, 0:-1]
        feat_target_output = feat_target[:, 1:]
        feat_target_input_len = [x-1 for x in feat_target_len]
        feat_target_mask = Variable(generate_seq_mask(seq_len=feat_target_input_len, 
            device=opts['gpu']))
        batch, dec_len, _ = feat_target_input.size()

        model_gen.reset()
        model_gen.train(train_step)
        model_gen.encode(feat_source, feat_source_len)
        list_dec_core = []
        for ii in range(dec_len) :
            _dec_core_ii, _ = model_gen.decode(feat_target_input[:, ii], feat_target_mask[:, ii] if opts['mask_dec'] else None)
            list_dec_core.append(_dec_core_ii)
            pass
        dec_core = torch.stack(list_dec_core,1)
        
        # calculate loss and update #
        loss_sup = criterion_recon(feat_target_output, dec_core,
                mask=None) # TODO : decide use mask or not #
        if train_step :
            opt.zero_grad()
            torch.nn.utils.clip_grad_norm(model_gen.parameters(), opts['grad_clip'])
            loss_sup.backward()
            opt.step()

        # log #
        if tfboard_writer is not None :
            tfboard_writer.add_scalar('loss/sup {}'.format(model_name), 
                    loss_sup.data.cpu().numpy(), niter)
            pass 
        pass

    def fn_critic(model_disc, feat_source, feat_len, train_step=True) :
        if not isinstance(feat_source, Variable) :
            feat_source = Variable(feat_source)
        model_disc.train(train_step)
        res = model_disc.forward(feat_source)
        return res
    
    def fn_train_gan(feat_real_a, feat_real_b, epoch=0, niter=0, tfboard_writer=None) :
        ### standard function for train normal GAN, least square GAN ###

        ### 1. update generative model gen_a & gen_b ###

        # generate fake feature & prediction given fake real feature #
        feat_fake_a = fn_generate(model_gen_b2a, feat_real_b, feat_len_real_b)
        feat_recon_b = fn_generate(model_gen_a2b, feat_fake_a, feat_len_real_b)

        feat_fake_b = fn_generate(model_gen_a2b, feat_real_a, feat_len_real_a)
        feat_recon_a = fn_generate(model_gen_b2a, feat_fake_b, feat_len_real_a)

        # create fake a & b and reconstruct their origin #
        pred_fake_a = fn_critic(model_disc_a, feat_fake_a, feat_len_real_b)
        pred_fake_b = fn_critic(model_disc_b, feat_fake_b, feat_len_real_a)
        
        # calculate loss (fake, target=True)
        loss_gan_gen_a = criterion_gan(pred_fake_a, target=True, mask=None)
        loss_gan_gen_b = criterion_gan(pred_fake_b, target=True, mask=None)

        # calculate loss reconstruction (cycle consistency loss)
        loss_recon_gen_a = criterion_recon(feat_recon_a, target=feat_real_a, mask=None)
        loss_recon_gen_b = criterion_recon(feat_recon_b, target=feat_real_b, mask=None)

        # calculate loss identity 
        loss_idt_gen_a = criterion_recon(feat_fake_a, target=feat_real_b, mask=None)
        loss_idt_gen_b = criterion_recon(feat_fake_b, target=feat_real_a, mask=None)

        loss_gen = (scheduler_coeff_gan.value * (loss_gan_gen_a + loss_gan_gen_b) + 
                opts['c_cycle'] * (loss_recon_gen_a + loss_recon_gen_b) +
                opts['c_idt'] * (loss_idt_gen_a + loss_idt_gen_b))

        # calculate backward & update generative model #
        if train_step :
            optim_gen_a2b.zero_grad()
            optim_gen_b2a.zero_grad()
            loss_gen.backward()
            optim_gen_a2b.step()
            optim_gen_b2a.step()
        
        ### 2. update discriminative model disc_a & disc_b #
        pred_real_a = fn_critic(model_disc_a, feat_real_a, feat_len_real_a)
        pred_real_b = fn_critic(model_disc_b, feat_real_b, feat_len_real_b)
        pred_fake_a = fn_critic(model_disc_a, feat_fake_a.detach(), feat_len_real_a)
        pred_fake_b = fn_critic(model_disc_b, feat_fake_b.detach(), feat_len_real_b)

        loss_gan_disc_a = criterion_gan(pred_real_a, target=True, mask=None) \
                + criterion_gan(pred_fake_a, target=False, mask=None)
        loss_gan_disc_b = criterion_gan(pred_real_b, target=True, mask=None) \
                + criterion_gan(pred_fake_b, target=False, mask=None)
        loss_disc = opts['c_gan'] * (loss_gan_disc_a + loss_gan_disc_b)

        # calculate backward & update generative model #
        if train_step :
            optim_disc_a.zero_grad()
            optim_disc_b.zero_grad()
            loss_disc.backward()
            optim_disc_a.step()
            optim_disc_b.step()

        # write information #
        if tfboard_writer is not None :
            tfboard_writer.add_scalar('loss/recon a', loss_recon_gen_a.data.cpu().numpy(), niter)
            tfboard_writer.add_scalar('loss/recon b', loss_recon_gen_b.data.cpu().numpy(), niter)
            tfboard_writer.add_scalar('loss/idt a', loss_idt_gen_a.data.cpu().numpy(), niter)
            tfboard_writer.add_scalar('loss/idt b', loss_idt_gen_b.data.cpu().numpy(), niter)
            tfboard_writer.add_scalar('loss/gan disc a', loss_gan_disc_a.data.cpu().numpy(), niter)
            tfboard_writer.add_scalar('loss/gan disc b', loss_gan_disc_b.data.cpu().numpy(), niter)
        pass

    #### SORT BY LENGTH - SortaGrad ####
    train_idx_a = feat_iterator.get_index_by_key(list_key_spk_a)
    train_idx_b = feat_iterator.get_index_by_key(list_key_spk_b)
    train_len_a = dict([(x, y) for x, y in zip(train_idx_a, feat_iterator.get_feat_length_by_key(list_key_spk_a))])
    train_len_b = dict([(x, y) for x, y in zip(train_idx_b, feat_iterator.get_feat_length_by_key(list_key_spk_b))])

    sorted_train_idx_a = sorted(train_idx_a, key=lambda x:train_len_a[x])
    sorted_train_idx_b = sorted(train_idx_b, key=lambda x:train_len_b[x])

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

    # save opts and create serializer fn  #
    json.dump(opts, open(os.path.join(opts['result'], 'script.opts'), 'w'), indent=4)
    def save_model(model, name, epoch) :
        torch.save(ModelSerializer.convert_param_to_cpu(model.state_dict()), os.path.join(opts['result'], 'model_{}_e{}.mdl'.format(name, epoch)))
        ModelSerializer.save_config(os.path.join(opts['result'], 'model_{}_e{}.cfg'.format(name, epoch)), model.get_config())
        pass

    tfboard_path = os.path.join(opts['result'], 'tf_'+datetime.now().replace(microsecond=0).isoformat())
    tfboard_writer = SummaryWriter(tfboard_path)

    # exclude cutoff #
    print("=====START=====")
    prev_dev_loss = 2**64
    best_dev_loss = 2**64
    niter = 0
    for ee in range(EPOCHS) :
        print('epoch {}'.format(ee))
        start_time = time.time()
       
        # choose standard training or bucket training #
        if opts['bucket'] :
            if opts['block'] == -1 :
                train_rr_a = iter_minibucket(sorted_train_idx_a, BATCHSIZE, 
                    shuffle=False if ee < opts['sortagrad'] else True, 
                    excludes=[])
                train_rr_b = iter_minibucket(sorted_train_idx_b, BATCHSIZE, 
                    shuffle=False if ee < opts['sortagrad'] else True, 
                    excludes=[])

            else :
                train_rr_a = iter_minibucket_block(sorted_train_idx_a, 
                    opts['block'], sorted_train_len, 
                    shuffle=False if ee < opts['sortagrad'] else True, 
                    pad=True, excludes=[])
                train_rr_b = iter_minibucket_block(sorted_train_idx_b, 
                    opts['block'], sorted_train_len, 
                    shuffle=False if ee < opts['sortagrad'] else True, 
                    pad=True, excludes=[])
        else :
            train_rr_a = iter_minibatches(sorted_train_idx_a, BATCHSIZE, 
                    shuffle=False if ee < opts['sortagrad'] else True, 
                    excludes=[])
            train_rr_b = iter_minibatches(sorted_train_idx_b, BATCHSIZE, 
                    shuffle=False if ee < opts['sortagrad'] else True, 
                    excludes=[])
        ###############################################
        train_rr_a = [sort_reverse(x, train_len_a) for x in train_rr_a] 
        train_rr_b = [sort_reverse(x, train_len_b) for x in train_rr_b] 
        ###############################################
        if len(train_rr_a) > len(train_rr_b) :
            pair_train_rr = (train_rr_a, itertools.cycle(train_rr_b))
            len_rr = len(train_rr_a)
        else :
            pair_train_rr = (itertools.cycle(train_rr_a), train_rr_b)
            len_rr = len(train_rr_b)

        # cycle dataset A & B #
        for set_name, (set_rr_a, set_rr_b), set_train_mode in [('train', pair_train_rr, True)] :
            for rr_a, rr_b in tqdm_wrapper(list(zip(set_rr_a, set_rr_b))) :
                # collect list of features #
                curr_feat_real_a_key = feat_iterator.get_key_by_index(rr_a)
                curr_feat_real_a = feat_iterator.get_feat_by_index(rr_a)

                curr_feat_real_b_key = feat_iterator.get_key_by_index(rr_b)
                curr_feat_real_b = feat_iterator.get_feat_by_index(rr_b)


                # batch data unpaired #
                feat_mat_real_a, feat_len_real_a = batch_speech(opts['gpu'],
                        curr_feat_real_a, feat_sil=feat_sil,
                        group=opts['group'], start_sil=1, end_sil=opts['pad_sil'])
                feat_mat_real_b, feat_len_real_b = batch_speech(opts['gpu'],
                        curr_feat_real_b, feat_sil=feat_sil,
                        group=opts['group'], start_sil=1, end_sil=opts['pad_sil'])

                feat_real_a = Variable(feat_mat_real_a)
                feat_real_b = Variable(feat_mat_real_b)
                ### special : paired data training ###
                if pair_info is not None :
                    try :
                        curr_feat_real_b_given_a = feat_iterator.get_feat_by_key(
                            [pair_info[x] for x in curr_feat_real_a_key]
                            )
                        curr_feat_real_a_given_b = feat_iterator.get_feat_by_key(
                            [pair_info[x] for x in curr_feat_real_b_key]
                            )
                    except KeyError :
                        import ipdb; ipdb.set_trace()
                    # batch data paired # 
                    feat_mat_real_b_given_a, feat_len_real_b_given_a = batch_speech(opts['gpu'],
                            curr_feat_real_b_given_a, feat_sil=feat_sil, 
                            group=opts['group'], start_sil=1, end_sil=opts['pad_sil'])
                    feat_mat_real_a_given_b, feat_len_real_a_given_b = batch_speech(opts['gpu'],
                            curr_feat_real_a_given_b, feat_sil=feat_sil, 
                            group=opts['group'], start_sil=1, end_sil=opts['pad_sil'])
                    feat_real_a_given_b = Variable(feat_mat_real_a_given_b)
                    feat_real_b_given_a = Variable(feat_mat_real_b_given_a)
                    # train paired model #
                    fn_generate_dynamic_supervised(model_gen_a2b, 
                            feat_real_a, feat_len_real_a, 
                            feat_real_b_given_a, feat_len_real_b_given_a,
                            train_step=True, model_name='a2a', 
                            tfboard_writer=tfboard_writer, niter=niter, 
                            opt=optim_gen_a2b)

                    fn_generate_dynamic_supervised(model_gen_b2a, 
                            feat_real_b, feat_len_real_b, 
                            feat_real_a_given_b, feat_len_real_a_given_b,
                            train_step=True, model_name='b2a',
                            tfboard_writer=tfboard_writer, niter=niter, 
                            opt=optim_gen_b2a)



                # train and evaluate gan #
                if opts['gan_type'] == 'none' :
                    pass # do nothing #
                elif opts['gan_type'] == 'lsgan' :
                    fn_train_gan(
                            feat_real_a=feat_real_a,
                            feat_real_b=feat_real_b,
                            epoch=ee, niter=niter,
                            tfboard_writer=tfboard_writer)
                else :
                    pass

                niter += 1 # global iteration counter #
                pass
            pass

        # info_header = ['set', 'loss', 'loss core', 'loss bern end', 'acc bern end']
        # info_table = []
        # logger.info("Epoch %d -- lrate %f -- time %.2fs"%(ee+1, opts['lrate'], time.time() - start_time))
        # for set_name in mloss.keys() :
            # mloss[set_name] /= mcount[set_name]
            # mloss_core[set_name] /= mcount[set_name]
            # mloss_bernend[set_name] /= mcount[set_name]
            # macc_bernend[set_name] /= mcount[set_name]
            # info_table.append([set_name, mloss[set_name], mloss_core[set_name], mloss_bernend[set_name], macc_bernend[set_name]])
        # logger.info('\n'+tab.tabulate(info_table, headers=info_header, floatfmt='.3f', tablefmt='rst'))

        # serialized best dev model #
        save_model(model_gen_a2b, 'gen_a2b', ee)
        save_model(model_gen_b2a, 'gen_b2a', ee)

        # increase step scheduler #
        scheduler_coeff_gan.step()
        if tfboard_writer is not None :
            tfboard_writer.add_scalar('coeff/coeff_gan', scheduler_coeff_gan.value, ee)
        pass

    pass
