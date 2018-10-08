#!/usr/bin/env python

import numpy
import json

import sys
import fileinput
from euterpe.config import constant
import argparse
from collections import OrderedDict

MAXVAL = 10**15
def parse() :
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cut', type=int, default=1, help='remove first n split (remove key)')
    parser.add_argument('-f', '--file', type=str, help='text file', nargs='+')
    return parser.parse_args()

def main():
    args = parse()
    cutidx = args.cut
    files = args.file
    for filename in files:
        print('Processing ' +filename)
        word_freqs = OrderedDict()
        with open(filename, 'r') as f:
            for line in f:
                words_in = line.strip().split()[cutidx:]
                for w in words_in:
                    if w not in word_freqs:
                        word_freqs[w] = 0
                    word_freqs[w] += 1
        words = list(word_freqs.keys())
        freqs = list(word_freqs.values())

        sorted_idx = numpy.argsort(freqs)
        sorted_words = [words[ii] for ii in sorted_idx[::-1]]

        worddict = OrderedDict()
        freqdict = OrderedDict()
        worddict[constant.UNK_WORD] = constant.UNK
        worddict[constant.BOS_WORD] = constant.BOS
        worddict[constant.EOS_WORD] = constant.EOS
        worddict[constant.PAD_WORD] = constant.PAD

        freqdict[constant.UNK_WORD] = MAXVAL
        freqdict[constant.BOS_WORD] = MAXVAL
        freqdict[constant.EOS_WORD] = MAXVAL
        freqdict[constant.PAD_WORD] = MAXVAL
        offset = len(worddict)
        for ii, ww in enumerate(sorted_words):
            if ww not in worddict :
                worddict[ww] = offset
                offset += 1
                freqdict[ww] = word_freqs[ww]

        with open('%s.dict'%filename, 'w') as f:
            json.dump(worddict, f, indent=2, ensure_ascii=False)
        with open('%s.freq'%filename, 'w') as f:
            json.dump(freqdict, f, indent=2, ensure_ascii=False)

        print('Done')

if __name__ == '__main__':
    main()
