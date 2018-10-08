import sys
import os
import argparse
import re
from euterpe.data.text_util.cleaners import english_cleaners

re_kv = re.compile('^([^\s]+) (.*)$', re.MULTILINE)

def parser() :
    args = argparse.ArgumentParser()
    args.add_argument('--text', type=str, help='raw text file <key> <text>')
    return args.parse_args()

if __name__ == '__main__':
    args = parser() 
    f = re_kv.findall(open(args.text).read())
    for k,v in f :
        print('{} {}'.format(k, english_cleaners(v)))  
    pass
