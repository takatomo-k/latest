"""
generator_s2t.py

generator for seq2seq model (e.g., speech -> text)

[+] beam search
[+] greedy search
"""

import torch
from torch.autograd import Variable
from torch.nn import functional as F
from torchev.utils.serializer import ModelSerializer
from torchev.utils.helper import tensorauto, torchauto, vars_index_select
from torchev.utils.mask_util import generate_seq_mask
from torchev.nn.utils.rnn import pad_sequence
from ..util.plot_attention import crop_attention_matrix
from ..config import constant

class Beam :
    def __init__(self, state, output=[], log_prob=0, score=0, coeff_lp=1.0) :
        """
        state : idx from model batch
        output : record for current output
        """
        self.state = state
        self.output = output
        self.log_prob = log_prob
        self.score = score 
        self.coeff_lp = coeff_lp
        self.active = True
        pass

    def len_penalty(self, hypo_len, alpha=1.0, const=5) :
        """
        Google NMT 
        lp(Y) = (5+|Y|)^alpha / (5+1)^alpha
        """
        return ((const + hypo_len)**alpha) / ((const+1)**alpha)

    def sample(self, state_t, log_prob_t, topk=5) :
        topk_log_prob_t, topk_log_prob_t_idx = log_prob_t.topk(topk, dim=0)
        topk_log_prob_t = topk_log_prob_t + self.log_prob.expand_as(topk_log_prob_t)
        topk_scores_t = topk_log_prob_t / self.len_penalty(len(self.output), alpha=self.coeff_lp)
        new_beams = []
        for ii in range(topk) :
            _beam = Beam(self.state+[state_t], self.output + [topk_log_prob_t_idx[ii]], 
                    topk_log_prob_t[ii], topk_scores_t[ii], self.coeff_lp)
            if _beam.output != [] and _beam.output[-1].data[0] == constant.EOS :
                _beam.active = False
            new_beams.append(_beam)
        return new_beams

    def __repr__(self) :
        return "[score : {}, output : {}, state : {}]".format(self.score.data[0], [x.data[0] for x in self.output], self.state)
    pass

class PoolBeam :
    def __init__(self, topk=5, nbest=1) :
        self.stack = []
        self.topk = topk
        assert nbest <= topk, 'nbest must be less or equal than beam size (topk)'
        self.nbest = nbest
        pass

    def add_beam(self, beam) :
        self.stack.append(beam)

    def get_active_beam(self) :
        return [x for x in self.stack if x.active]

    def get_finished_beam(self) :
        return [x for x in self.stack if not x.active]
    
    def is_finished(self) :
        # check if nbest all not active #
        return all([not x.active for x in self.stack[0:self.nbest]])

    def step(self, state_t, log_prob_t) :
        # get previous active beam only #
        active_beams = self.get_active_beam()
        assert log_prob_t.size(0) == len(active_beams)
        new_candidate_beams = []
        # ask active beam to sample new beam based on current score #
        for ii in range(len(active_beams)) :
            new_candidate_beams.extend(active_beams[ii].sample(state_t[ii], log_prob_t[ii], self.topk))
        # remove current active beams from stack, keep only finished beam to be compared  #
        # why? because if beam is active, it will be replaced by new candidate #
        self.stack = [x for x in self.stack if not x.active]
        # gather new candidate with all previous candidate #
        self.stack.extend(new_candidate_beams) 
        # sort by largest score & prune #
        self.sort()
        pass

    def sort(self) :
        # sort by largest score & prune #
        self.stack = sorted(self.stack, key=lambda x : x.score.data[0], reverse=True)[0:self.topk]

    
    @staticmethod
    def combine_state_and_input(device, beams, prev_state) :
        state = vars_index_select(prev_state, 0, Variable(torchauto(device).LongTensor([x.state[-1] for x in beams]))) # take latest state_idx #
        input = torch.cat([x.output[-1] for x in beams])
        return state, input
        pass

    def __repr__(self) :
        return ','.join(map(str, self.stack))

def eval_gen_text_quality(texts, text_len, att_mat) :
    # TODO improve criterion
    batch_size = len(text_len)
    quality = [None for _ in range(batch_size)]
    for bb in range(batch_size) :
        if text_len[bb] < 0 :
            quality[bb] = 0
        else :
            quality[bb] = 1
    return quality

def beam_search(model, src_mat, src_len, map_text2idx, max_target, kbeam=5, coeff_lp=1.0) :
    transcription, transcription_len, att_mat = beam_search_torch(model, src_mat, src_len, map_text2idx, max_target, kbeam=kbeam, coeff_lp=coeff_lp)
    return convert_tensor_to_seq_list(transcription, transcription_len), transcription_len, \
            crop_attention_matrix(att_mat.data.cpu().numpy(), transcription_len, model.dec_att_lyr.ctx_len)

def beam_search_nbest(model, src_mat, src_len, map_text2idx, max_target, kbeam=5, coeff_lp=1.0, nbest=1) :
    raise NotImplementedError("need to recheck this method correctness")
    transcription, transcription_len, att_mat = beam_search_torch(model, src_mat, src_len, map_text2idx, max_target, kbeam=kbeam, coeff_lp=coeff_lp, nbest=nbest)
    transcription_text = []
    for bb in range(len(transcription)) :
        _text_bb = convert_tensor_to_seq_list(transcription[bb], transcription_len[bb]) 
        transcription_text.append(_text_bb)
    return transcription_text, transcription_len, att_mat
    pass

def beam_search_torch(model, src_mat, src_len, map_text2idx, max_target, kbeam=5, coeff_lp=1.0, nbest=None) :
    """
    http://opennmt.net/OpenNMT/translation/beam_search/

    nbest : 
        if None -> took only top-1
            return transcription (batch x words) (list of words)
        if n >= 1 -> took top-n
            return transcription (batch x nbest x words) (list of list of words)
    """
    if not isinstance(src_mat, Variable) :
        src_mat = Variable(src_mat)
    batch, max_src_len = src_mat.size()[0:2]
    # run encoder to get context and mask #
    model.eval()
    model.reset()
    model.encode(src_mat, src_len)
    enc_ctx = model.dec_att_lyr.ctx
    enc_ctx_mask = model.dec_att_lyr.ctx_mask
    # expand encoder & encoder mask for each beam #
    max_src_len = enc_ctx.size(1)

    pool_beams = []
    curr_input = Variable(torchauto(model).LongTensor([constant.BOS for _ in range(batch)]))

    # start by adding 1 beam per batch #
    for bb in range(batch) :
        if nbest is None :
            pool_beam = PoolBeam(topk=kbeam, nbest=1)
        else :
            pool_beam = PoolBeam(topk=kbeam, nbest=nbest)

        pool_beam.add_beam(Beam([], output=[],
            log_prob = Variable(torchauto(model).FloatTensor([0])), 
            score = Variable(torchauto(model).FloatTensor([0])), 
            coeff_lp=coeff_lp))
        pool_beams.append(pool_beam)
    
    active_batch = [1 for x in range(batch)]

    def local2global_encoder(active_batch) :
        """
        function to convert number of active beam beam for each batch into list of encoder side index
        """
        return sum([[bb for _ in range(active_batch[bb])] for bb in range(batch)], [])

    global_state_index = Variable(torchauto(model).LongTensor(local2global_encoder(active_batch)))
    list_pre_softmax = []
    list_att_mat = []
    for tt in range(max_target) :
        pre_softmax, dec_output = model.decode(curr_input)
        list_pre_softmax.append(pre_softmax)
        list_att_mat.append(dec_output['att_output']['p_ctx'])
        log_prob = F.log_softmax(pre_softmax, dim=-1)
        new_beams = []
        start, end = [], []
        for bb in range(batch) :
            start.append(0 if bb == 0 else end[bb-1])
            end.append(start[-1]+active_batch[bb])
        for bb in range(batch) :
            if pool_beams[bb].is_finished() or active_batch[bb] == 0:
                continue
            # distribute softmax calculation to each beam #
            log_prob_bb = log_prob[start[bb]:end[bb]]
            pool_beams[bb].step(list(range(start[bb], end[bb])), log_prob_bb)
            # renew active_batch and finished_batch information #
            _tmp_active_beam = pool_beams[bb].get_active_beam()
            active_batch[bb] = len(_tmp_active_beam) if not pool_beams[bb].is_finished() else 0
            if not pool_beams[bb].is_finished() :
                # if this pool has not finish, continue put new beam #
                new_beams.extend(_tmp_active_beam)

        if all(x.is_finished() for x in pool_beams) :
            # all batch finished #
            break
        # clear up and get new input + state #
        curr_state, curr_input = PoolBeam.combine_state_and_input(model, new_beams, model.state)
        
        global_state_index = Variable(torchauto(model).LongTensor(local2global_encoder(active_batch)))
        # model ctx & ctx mask #
        model.dec_att_lyr.ctx = enc_ctx.index_select(0, global_state_index)
        model.dec_att_lyr.ctx_mask = enc_ctx_mask.index_select(0, global_state_index)
        # model state #
        model.state = curr_state
        pass
    # gather all top of stack #
    if nbest is None :
        transcription = []
        transcription_len = []
        att_mat = []
        map_idx2text = dict([y, x] for (x, y) in map_text2idx.items())
        for bb in range(batch) :
            best_beam = pool_beams[bb].stack[0]
            # gather transcription #
            transcription.append(torch.cat(best_beam.output))
            transcription_len.append(len(best_beam.output) if pool_beams[bb].is_finished() else -1)
            # gather attention matrix #
            att_mat_bb = []
            for ii in range(len(best_beam.state)) :
                att_mat_bb.append(list_att_mat[ii][best_beam.state[ii]])
            att_mat_bb = torch.stack(att_mat_bb, 0)
            att_mat.append(att_mat_bb)
        att_mat = pad_sequence(att_mat, batch_first=True)
        return transcription, transcription_len, att_mat
    else :
        transcription = []
        transcription_len = []
        att_mat = []
        map_idx2text = dict([y, x] for (x, y) in map_text2idx.items())
        for bb in range(batch) :
            transcription.append([])
            transcription_len.append([])
            att_mat.append([])
            for nn in range(nbest) :
                best_beam = pool_beams[bb].stack[nn]
                # gather transcription #
                transcription[bb].append(torch.cat(best_beam.output))
                transcription_len[bb].append(len(best_beam.output) if pool_beams[bb].is_finished() else -1)
                # gather attention matrix #
                att_mat_bb_nn = []
                for ii in range(len(best_beam.state)) :
                    att_mat_bb_nn.append(list_att_mat[ii][best_beam.state[ii]])
                att_mat_bb_nn = torch.stack(att_mat_bb_nn, 0)
                att_mat[bb].append(att_mat_bb_nn)
        return transcription, transcription_len, att_mat
    pass

def greedy_search(model, src_mat, src_len, map_text2idx, max_target) :
    transcription, transcription_len, att_mat = greedy_search_torch(model, src_mat, src_len, map_text2idx, max_target)
    return convert_tensor_to_seq_list(transcription, transcription_len), transcription_len, crop_attention_matrix(att_mat.data.cpu().numpy(), transcription_len, model.dec_att_lyr.ctx_len)

def greedy_search_torch(model, src_mat, src_len, map_text2idx, max_target) :
    if not isinstance(src_mat, Variable) :
        src_mat = Variable(src_mat)

    batch = src_mat.size(0)
    model.eval()
    model.reset()
    model.encode(src_mat, src_len)

    prev_label = Variable(torchauto(model).LongTensor([map_text2idx[constant.BOS_WORD] for _ in range(batch)]))
    transcription = []
    transcription_len = [-1 for _ in range(batch)]
    att_mat = []
    for tt in range(max_target) :
        pre_softmax, dec_res = model.decode(prev_label)
        max_id = pre_softmax.max(1)[1]
        transcription.append(max_id)
        att_mat.append(dec_res['att_output']['p_ctx'])
        for ii in range(batch) :
            if transcription_len[ii] == -1 :
                if max_id.data[ii] == map_text2idx[constant.EOS_WORD] :
                    transcription_len[ii] = tt+1
        if all([ii != -1 for ii in transcription_len]) :
            # finish #
            break
        prev_label = max_id
        pass

    # concat across all timestep #
    transcription = torch.stack(transcription, 1) # batch x seq_len #
    att_mat = torch.stack(att_mat, 1) # batch x seq_len x enc_len #
    
    return transcription, transcription_len, att_mat
    pass

def convert_tensor_to_seq_list(batch_mat, seq_len, exclude_eos=True) :
    result = []
    for ii in range(len(seq_len)) :
        curr_result = batch_mat[ii].data.cpu().numpy().tolist()
        if seq_len[ii] != -1 :
            curr_result = curr_result[0:seq_len[ii] - (1 if exclude_eos else 0)]
        result.append(curr_result)
    return result

def teacher_forcing_torch(model, src_mat, src_len, tgt_mat, tgt_len, map_text2idx, max_target) :
    if not isinstance(src_mat, Variable) :
        src_mat = Variable(src_mat)
    if not isinstance(tgt_mat, Variable) :
        tgt_mat = Variable(tgt_mat)
    tgt_input = tgt_mat[:, 0:-1]
    tgt_output = tgt_mat[: 1:]
    model.eval()
    model.reset()
    model.encode(src_mat, src_len)
    batch, dec_len = tgt_input.size()
    list_pre_softmax = []
    list_att_mat = []
    for tt in range(dec_len) :
        pre_softmax_tt, dec_res_tt = model.decode(tgt_input[:, tt])
        list_pre_softmax.append(pre_softmax_tt)
        list_att_mat.append(dec_res_tt['att_output']['p_ctx'])
    pre_softmax = torch.stack(list_pre_softmax, 1) # batch x dec_len x nclass #
    att_mat = torch.stack(list_att_mat, 1) # batch x dec_len x enc_len #
    tgt_output_len = [x-1 for x in tgt_len] # exclude <bos>, include <eos>

    # TODO : apply mask delete ?
    # dec_mask = Variable(generate_seq_mask(tgt_output_len, device=model))
    # att_mat = att_mat * dec_mask
    return pre_softmax, tgt_output_len, att_mat

def teacher_forcing(model, src_mat, src_len, tgt_mat, tgt_len, map_text2idx, max_target, exclude_eos=False) :
    pre_softmax, transcription_len, att_mat = teacher_forcing_torch(model, src_mat, src_len, tgt_mat, tgt_len, map_text2idx, max_target)
    return convert_tensor_to_seq_list(pre_softmax, transcription_len, exclude_eos=exclude_eos), transcription_len,\
            crop_attention_matrix(att_mat.data.cpu().numpy(), transcription_len, model.dec_att_lyr.ctx_len)

# TODO : monte-carlo sampler #
