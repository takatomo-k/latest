from ..math_util import softmax, log_softmax
import numpy as np
def test_softmax() :
    mat = [[1., 2., 3.], [5., 9., -3.]]
    mat = np.array(mat)
    result = softmax(mat)
    assert np.allclose(result.sum(axis=1), 1.)

def test_log_softmax() :
    mat = [[1., 2., 3.], [5., 9., -3.]]
    mat = np.array(mat)
    result = log_softmax(mat)
    assert np.allclose(np.exp(result).sum(axis=1), 1.)
