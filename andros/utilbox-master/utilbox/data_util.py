import numpy as np

def pad_idx(indices, batchsize) :
    if batchsize == 0 :
        return indices
    assert len(indices) >= batchsize, "batchsize must be less or equal than number of data sample"
    if len(indices) % batchsize == 0 :
        return indices
    num_remaining = batchsize - (len(indices) % batchsize)
    return indices+indices[0:num_remaining]
    pass

def one_hot(x, m=None) :
    if m is None :
        m = np.max(x)+1
    mat = np.zeros((len(x), m))
    mat[np.arange(len(x)), x] = 1.0
    return mat

def stat_array(x, axis=0) :
    return {'min':x.min(axis=axis), 'max':x.max(axis=axis), 'mean':x.mean(axis=axis), 'std':x.std(axis=axis)}

class IncrementalMax :
    def __init__(self, axis=0) :
        self.axis = axis
        self.stat_max = None
        pass
    
    def update(self, x) :
        new_max = np.max(x, axis=self.axis)
        if self.stat_max is not None :
            new_max = np.max([self.stat_max, new_max], axis=0)
        self.stat_max = new_max
        pass

class IncrementalMin :
    def __init__(self, axis=0) :
        self.axis = axis
        self.stat_min = None
        pass
    
    def update(self, x) :
        new_min = np.min(x, axis=self.axis)
        if self.stat_min is not None :
            new_min = np.min([self.stat_min, new_min], axis=0)
        self.stat_min = new_min
        pass


# DATA ITERATOR #
def iter_minibatches(indices, batchsize, shuffle=True, pad=False, excludes=None):
    """
    Args:
        datasize : total number of data or list of indices
        batchsize : mini-batchsize
        shuffle :
        use_padding : pad the dataset if dataset can't divided by batchsize equally

    Return :
        list of index for current epoch (randomized or not depends on shuffle)
    """
    if isinstance(indices, list) :
        indices = indices
    elif isinstance(indices, int) :
        indices = list(range(indices))
    if excludes is not None :
        indices = [x for x in indices if x not in excludes]
    if shuffle:
        np.random.shuffle(indices)

    if pad :
        indices = pad_idx(indices, batchsize)

    for ii in range(0, len(indices), batchsize):
        yield indices[ii:ii + batchsize]
    pass

def iter_minibucket(indices, batchsize, shuffle=True, excludes=None) :
    """
    Iterate bucket of index for efficient different sequence training
    Notes : returned bucket must be retrieved on sorted index
    Example :

    x = [datasize x seqlen x ndim]
    sidx = np.argsort(map(len, x))
    for rr in iter_minibucket(datasize, batchsize) :
        curr_x = [x[sidx[ii]] for ii in rr] # get sorted index and retrieve x
    """
    if isinstance(indices, list) :
        indices = indices
    elif isinstance(indices, int) :
        indices = list(range(indices))
    if excludes is not None :
        indices = [x for x in indices if x not in excludes]
    datasize = len(indices)
    indices = [indices[ii:ii+batchsize] for ii in range(0, datasize, batchsize)]
    if shuffle :
        np.random.shuffle(indices)
    for ii in range(0, len(indices)) :
        yield indices[ii]
    pass

def iter_minibucket_block(indices, blocksize, seqlen, pad_block=False, shuffle=True, excludes=None) :
    """
    Iterate bucket of index for efficient different sequence training
        example : if pivot_seqlen is 10, and batchsize is 5, we will keep the block < 50
        if batch now has seqlen(5, 10, 4, 5)
    """
    # V1 : seqlen must be sorted #
    # assert sorted(seqlen) == seqlen, "seqlen must be sorted incrementally"
    # V2 : automatically reorder indices wrt. sorqlen #
    assert max(seqlen) <= blocksize, "all seqlen must be smaller than blocksize, max(seqlen): {}".format(max(seqlen))
    assert len(seqlen) == len(indices) 
    if isinstance(indices, list) :
        indices = indices
    elif isinstance(indices, int) :
        indices = list(range(indices))
    if excludes is not None :
        new_indices = []
        new_seqlen = []
        for ii in range(len(indices)) :
            if indices[ii] not in excludes :
                new_indices.append(indices[ii])
                new_seqlen.append(seqlen[ii])
        indices = new_indices
        seqlen = new_seqlen
    assert len(seqlen) == len(indices)

    # reorder #
    sorted_idx = np.argsort(seqlen).tolist()
    indices = [indices[ii] for ii in sorted_idx]
    seqlen = [seqlen[ii] for ii in sorted_idx]
    assert np.all(np.array(sorted(seqlen)) == np.array(seqlen)), "seqlen must be sorted incrementally"

    curr_index = []
    final_index = []
    if not pad_block :
        curr_index_len = 0
        for ii in range(len(indices)) :
            # if current + candidate > blocksize, flush memory #
            if curr_index_len+seqlen[ii] > blocksize :  
                final_index.append(curr_index)
                curr_index, curr_index_len = [], 0
            curr_index.append(indices[ii])
            curr_index_len += seqlen[ii]
            pass
    else :
        for ii in range(len(indices)) :
            if seqlen[ii] * (len(curr_index)+1) > blocksize :
                final_index.append(curr_index)
                curr_index = []
            curr_index.append(indices[ii])
    if len(curr_index) != 0 :
        final_index.append(curr_index)
    if shuffle :
        np.random.shuffle(final_index)
    return final_index

def context_window(seq, size=5) :
    """
    Create list of overlap context window given sequence
    Primary used in signal processing

    Args :
        seq : list of frames [time x n-dim vector]
        size : range context window to the left and right (total frame = 1 + size * 2)
    """
    total = 1 + size*2
    ndim = seq.shape[1]
    result = np.zeros((seq.shape[0], seq.shape[1]*(total)))
    mid = size
    for ii in range(len(seq)) :
        left_seq = max(ii-size, 0)
        right_seq = min(ii+size, len(seq)-1)
        left_con = left_seq-(ii-size)
        right_con = total-(ii+size-right_seq)
        # print left_seq, right_seq
        # print left_con, right_con
        result[ii][left_con*ndim:right_con*ndim] = seq[left_seq:right_seq+1].flatten()
    return result
    pass

"""
io utils
"""

#
# # DATA ITERATOR #
# def iter_semi_minibatches(inputs_lbl, targets_lbl, inputs_unlbl, lbl_batchsize, unlbl_batchsize, shuffle=True, use_padding=True):
#     # ITERATE UNTIL BOTH PORTION PROCESSED #
#     assert len(inputs_lbl) == len(targets_lbl)
#     indices_lbl = np.arange(len(inputs_lbl))
#     indices_unlbl = np.arange(len(inputs_unlbl))
#
#     if shuffle:
#         np.random.shuffle(indices_lbl)
#         np.random.shuffle(indices_unlbl)
#
#     if use_padding:
#         indices_lbl = pad_idx(indices_lbl, lbl_batchsize)
#         indices_unlbl = pad_idx(indices_unlbl, unlbl_batchsize)
#
#     if lbl_batchsize == 0 :
#         num_batch_lbl = 0
#     else :
#         num_batch_lbl = (len(inputs_lbl)  + lbl_batchsize- 1) / lbl_batchsize
#     if unlbl_batchsize == 0 :
#         num_batch_unlbl = 0
#     else :
#         num_batch_unlbl = (len(inputs_unlbl) + unlbl_batchsize - 1) / unlbl_batchsize
#     pivot_num_batch = max(num_batch_lbl, num_batch_unlbl)
#     cidx_lbl = 0
#     cidx_unlbl = 0
#     for tt in range(pivot_num_batch) :
#         start_lbl = cidx_lbl*lbl_batchsize
#         end_lbl = (cidx_lbl+1)*lbl_batchsize
#         start_unlbl = cidx_unlbl*unlbl_batchsize
#         end_unlbl = (cidx_unlbl+1)*unlbl_batchsize
#         cidx_lbl += 1
#         cidx_unlbl += 1
#         if cidx_lbl >= num_batch_lbl :
#             cidx_lbl = 0
#         if cidx_unlbl >= num_batch_unlbl :
#             cidx_unlbl = 0
#         excerpt_lbl = indices_lbl[start_lbl:end_lbl]
#         excerpt_unlbl = indices_unlbl[start_unlbl:end_unlbl]
#         yield inputs_lbl[excerpt_lbl], targets_lbl[excerpt_lbl], inputs_unlbl[excerpt_unlbl]
#     pass
