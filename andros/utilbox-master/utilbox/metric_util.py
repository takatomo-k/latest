import numpy as np
import warnings
INF = 2**64
def np_levenshtein_distance(hypo, ref, count=True) :
    """
    np_levenshtein_distance 
    This implementation is slower than python list
    Don't use it
    """
    cache = np.zeros((len(ref)+1, len(hypo)+1))

    # init #
    cache[:, 0] = list(range(len(ref)+1))
    cache[0, :] = list(range(len(hypo)+1))

    for ii in range(1, len(ref)+1) :
        for jj in range(1, len(hypo)+1) :
            if ref[ii-1] == hypo[jj-1] :
                cache[ii, jj] = cache[ii-1, jj-1]
            else :
                _sub = cache[ii-1, jj-1]+1
                _del = cache[ii, jj-1]+1
                _ins = cache[ii-1, jj]+1
                cache[ii, jj] = min(_sub, _ins, _del)
            pass
        pass
    if count :
        ii, jj = len(ref), len(hypo)
        cnt_sub, cnt_del, cnt_ins = 0., 0., 0.
        while ii != 0 or jj != 0 :
            _cand_sub = cache[ii-1, jj-1] if ii-1 >= 0 and jj-1 >= 0 else INF
            _cand_ins = cache[ii-1, jj] if ii-1 >= 0 else INF
            _cand_del = cache[ii, jj-1] if jj-1 >= 0 else INF
            _best_cand = min(_cand_sub, _cand_ins, _cand_del)
            if _best_cand == _cand_sub :
                if cache[ii, jj] > _best_cand :
                    cnt_sub += 1
                ii, jj = ii-1, jj-1
            elif _best_cand == _cand_ins :
                cnt_ins += 1
                ii, jj = ii-1, jj
            elif _best_cand == _cand_del :
                cnt_del += 1
                ii, jj = ii, jj-1
            pass
        assert cache[-1, -1] == cnt_sub+cnt_ins+cnt_del
        return cache[-1, -1], cnt_sub, cnt_ins, cnt_del
        pass
    return cache[-1][-1]    
    pass

def levenshtein_distance(hypo, ref, count=True) :

    cache = [[0 for _ in range(len(hypo)+1)] for _ in range(len(ref)+1)]

    # init #
    for ii in range(len(ref)+1) :
        cache[ii][0] = ii
    for ii in range(len(hypo)+1) :
        cache[0][ii] = ii

    for ii in range(1, len(ref)+1) :
        for jj in range(1, len(hypo)+1) :
            if ref[ii-1] == hypo[jj-1] :
                cache[ii][jj] = cache[ii-1][jj-1]
            else :
                _sub = cache[ii-1][jj-1]+1
                _del = cache[ii][jj-1]+1
                _ins = cache[ii-1][jj]+1
                cache[ii][jj] = min(_sub, _ins, _del)
            pass
        pass
    if count :
        ii, jj = len(ref), len(hypo)
        cnt_sub, cnt_del, cnt_ins = 0., 0., 0.
        while ii > 0 or jj > 0 :
            _cand_sub = cache[ii-1][jj-1] if ii-1 >= 0 and jj-1 >= 0 else INF
            _cand_ins = cache[ii-1][jj] if ii-1 >= 0 else INF
            _cand_del = cache[ii][jj-1] if jj-1 >= 0 else INF
            _best_cand = min(_cand_sub, _cand_ins, _cand_del)
            if _best_cand == _cand_sub :
                if cache[ii][jj] > _best_cand :
                    cnt_sub += 1
                ii, jj = ii-1, jj-1
            elif _best_cand == _cand_ins :
                cnt_ins += 1
                ii, jj = ii-1, jj
            elif _best_cand == _cand_del :
                cnt_del += 1
                ii, jj = ii, jj-1
            pass
        assert cache[-1][-1] == cnt_sub+cnt_ins+cnt_del
        return cache[-1][-1], cnt_sub, cnt_ins, cnt_del
        pass
    return cache[-1][-1]
    pass

edit_distance = levenshtein_distance


############################
### DYNAMIC TIME WARPING ###
############################
def euclid_dist(a, b) :
    return np.sum((np.array(a) - np.array(b))**2)

def dynamic_time_warping(hypo, ref, fn_dist=euclid_dist, normalize=True) :
    
    cache = [[0 for _ in range(len(hypo)+1)] for _ in range(len(ref)+1)]

    # init #
    for ii in range(1, len(ref)+1) :
        cache[ii][0] = INF
    for ii in range(1, len(hypo)+1) :
        cache[0][ii] = INF

    for ii in range(1, len(ref)+1) :
        for jj in range(1, len(hypo)+1) :
            cost = fn_dist(ref[ii-1], hypo[jj-1])
            _sub = cache[ii-1][jj-1]
            _del = cache[ii][jj-1]
            _ins = cache[ii-1][jj]
            cache[ii][jj] = cost + min(_sub, _ins, _del)
            pass
        pass
    coeff = 1.0
    if normalize :
        coeff /= float(len(hypo) + len(ref))
    return cache[-1][-1] * coeff
    pass

if __name__ == '__main__' :
    ref = list("She is so beautiful")
    hypo = list("S i sho bzautifulllll")
    hypo2 = list("She is so bztiful")
    print((edit_distance(hypo, ref)))

    ref = list("She is so beautiful")
    hypo = list("S i sho bzautifulllll")
    hypo2 = list("She is so bztiful")
    print((dynamic_time_warping(hypo, ref, lambda x, y : 0 if x == y else 1)))

    ref = [0, 0, 1, 1, 2, 4, 2, 1, 2, 0]
    hypo = [1, 1, 1, 2, 2, 2, 2, 3, 2, 0]
    print((dynamic_time_warping(hypo, ref, lambda x, y : abs(x - y)**0.5)))
    pass
