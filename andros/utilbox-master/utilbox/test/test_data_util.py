import pytest
import numpy as np

from ..data_util import pad_idx, \
        iter_minibatches, iter_minibucket, \
        iter_minibucket_block, one_hot, \
        IncrementalMax, IncrementalMin

class TestPadIdx :
    def test_pad_idx(self) :
        INDICES = [1,2,3,4]
        BATCHSIZE = 3
        ACTUAL = [1,2,3,4,1,2]
        res = pad_idx(INDICES, BATCHSIZE)
        assert np.all(ACTUAL == res)
        pass

    def test_pad_idx_b0(self) :
        INDICES = [1,2,3,4]
        BATCHSIZE = 0
        ACTUAL = [1,2,3,4]
        res = pad_idx(INDICES, BATCHSIZE)
        assert np.all(ACTUAL == res)
        pass

    def test_pad_idx_mod0(self) :
        INDICES = [1,2,3,4]
        BATCHSIZE = 4
        ACTUAL = [1,2,3,4]
        res = pad_idx(INDICES, BATCHSIZE)
        assert np.all(ACTUAL == res)
        pass

class TestIterMinibatches :
    def test_iter_minibatches_noshuffle(self) :
        DATASIZE = 6
        BATCHSIZE = 4
        ACTUAL = [[0,1,2,3],[4,5,0,1]]

        res = list(iter_minibatches(DATASIZE, BATCHSIZE, False, True))
        assert np.all(ACTUAL == res)
    
    def test_iter_minibatches_noshuffle_2(self) :
        DATASIZE = 6
        BATCHSIZE = 4
        ACTUAL = [[0,1,2,3],[4,5,0,1]]

        res = list(iter_minibatches(DATASIZE, BATCHSIZE, True, True))
        counter = [0 for _ in range(8)]
        for ii in range(len(res)) :
            for iii in res[ii] :
                counter[iii] += 1
        one = 0
        two = 0
        for ii in range(len(counter)) :
            one += counter[ii] == 1
            two += counter[ii] == 2
        assert one == 4
        assert two == 2
    pass

class TestIterBucket :
    def test_iter_minibucket_noshuffle(self) :
        DATASIZE = 8
        BATCHSIZE = 3
        ACTUAL = [np.r_[0,1,2],np.r_[3,4,5],np.r_[6,7]]
        res = list(iter_minibucket(DATASIZE, BATCHSIZE, False))
        for x,y in zip(ACTUAL, res) :
            assert np.all(x == y)

class TestOneHot :
    def test_onehot(self) :
        LBL = [5,2,1]
        ACTUAL = np.zeros((len(LBL), 6))
        ACTUAL[np.arange(len(LBL)), np.array(LBL)] = 1.0
        res = one_hot(LBL)
        assert np.all(ACTUAL == res)
        pass

class TestIterMinibatchesBlock :
    def test_iter_minibatches_block_noshuffle(self) :
        BLOCKSIZE = 10
        SEQLEN = [1,2,3,4,4,7,9]
        DATASIZE = len(SEQLEN)

        res = list(iter_minibucket_block(DATASIZE, BLOCKSIZE, SEQLEN, False, False))
        for item in res :
            print(item)
            assert sum(item) <= BLOCKSIZE
        res = list(iter_minibucket_block(DATASIZE, BLOCKSIZE, SEQLEN, True, False))
        for item in res :
            print(item)
            assert max(item) * len(item) <= BLOCKSIZE
    
    def test_iter_minibatches_block_noshuffle_error(self) :
        BLOCKSIZE = 10
        SEQLEN = [1,2,3,4,4,7,11]
        DATASIZE = len(SEQLEN)
        with pytest.raises(Exception) :
            res = list(iter_minibucket_block(DATASIZE, BLOCKSIZE, SEQLEN, False, False))

class TestIncrementalMax :
    def test_incremental_max(self) :
        x = np.random.rand(10, 5)
        incmax = IncrementalMax()
        for ii in range(0, 10, 2) :
            incmax.update(x[ii:ii+2])
        assert np.all(incmax.stat_max == x.max(axis=0))

class TestIncrementalMin :
    def test_incremental_min(self) :
        x = np.random.rand(10, 5)
        incmax = IncrementalMin()
        for ii in range(0, 10, 2) :
            incmax.update(x[ii:ii+2])
        assert np.all(incmax.stat_min == x.min(axis=0))
