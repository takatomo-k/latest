import sys
import warnings

def printe(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def printw(*args, **kwargs):
    warnings.warn(*args, **kwargs)
