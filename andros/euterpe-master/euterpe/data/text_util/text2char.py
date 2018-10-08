import sys
import re
import argparse

re_kv = re.compile('(^[^\s]+) (.+)$', re.MULTILINE)

def parse() :
    args = argparse.ArgumentParser()
    args.add_argument('--text', type=str, help='text path file')
    return args.parse_args()

def text2char(text) :
    text = re.sub('\s+', '*', text)
    text = re.sub('_', ' ', text)
    text = re.sub('"', "'", text)
    text = ' '.join(list(text))
    text = re.sub('\*', '<spc>', text)
    return text

if __name__ == '__main__':
    args = parse()
    f = re_kv.findall(open(args.text).read())
    for k,v in f :
        print('{} {}'.format(k, text2char(v)))
