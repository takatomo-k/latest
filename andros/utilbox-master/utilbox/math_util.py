import numpy as np
from scipy.misc import logsumexp

def assert_nan(value) :
    if np.any(np.isnan(value)) :
        raise ValueError("NaN detected")

def dimshuffle(mat, pattern) :
    pattern_wo_x = [x for x in list(pattern) if x != 'x' and x != 'X']
    print(pattern_wo_x)
    pos_x = [x[0] for x in [x for x in enumerate(pattern) if x[1] == 'x' or x[1] == 'X']]
    mat = np.transpose(mat, axes=pattern_wo_x)
    for pos in pos_x :
        mat = np.expand_dims(mat, axis=pos)
    return mat
    pass

def softmax(mat) :
    assert mat.ndim == 2, "Input must be 2-dim"
    mat = mat.T
    e_mat = np.exp(mat - np.max(mat, axis=0))
    prob_mat = e_mat / e_mat.sum(axis=0)
    return prob_mat.T
    pass

def log_softmax(mat) :
    assert mat.ndim == 2, "Input must be 2-dim"
    mat = mat.T
    e_mat = mat - np.max(mat, axis=0)
    log_prob_mat = e_mat - logsumexp(e_mat, axis=0)
    return log_prob_mat.T
    pass

