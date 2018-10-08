import argparse

from utilbox.regex_util import regex_key_val

from euterpe.common.loader import DataLoader

def parse() :
    parser = argparse.ArgumentParser()
    parser.add_argument('--kv', type=str)
    parser.add_argument('--set', type=str)
    return parser.parse_args()
    pass

if __name__ == '__main__':
    args = parse()    
    list_kv = DataLoader._read_key_val(args.kv)
    list_set = DataLoader._read_key(args.set)
    list_subset = DataLoader._subset_data(list_kv, list_set)
    for k, v in list_subset :
        print('{} {}'.format(k, v))
    pass
