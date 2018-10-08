import sys
import re
import argparse
import phonemizer

re_kv = re.compile('(^[^\s]+) (.+)$', re.MULTILINE)
char_sep = phonemizer.seperator.Seperator('<spc>', '', ' ')

def parse() :
    args = argparse.ArgumentParser()
    args.add_argument('--text', type=str, help='text path file')
    return args.parse_args()

def text2phoneme(text) :
    text = phonemizer.phonemize(text, seperator=char_sep)
    return text

if __name__ == '__main__':
    args = parse()
    f = re_kv.findall(open(args.text).read())
    for k,v in f :
        print('{} {}'.format(k, text2phoneme(v)))
