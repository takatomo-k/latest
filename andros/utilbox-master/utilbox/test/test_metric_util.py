import pytest
from ..metric_util import levenshtein_distance, edit_distance

def test_levenshtein_dist() :
    hyps = list("spartan")
    tgts = list("part")
    print(hyps, tgts)
    (total, a, b, c) = levenshtein_distance(hyps, tgts)
    assert a+b+c == total == 3
    (total, a, b, c) = edit_distance(tgts, hyps)
    assert a+b+c == total == 3
    pass

def test_levenshtein_dist() :
    hyps = list("spartan")
    tgts = list("")
    print(hyps, tgts)
    (total, a, b, c) = levenshtein_distance(hyps, tgts)
    assert a+b+c == total == 7 
    (total, a, b, c) = edit_distance(tgts, hyps)
    assert a+b+c == total == 7
    pass
