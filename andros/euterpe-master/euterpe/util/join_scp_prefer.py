from utilbox.regex_util import regex_key_val
import argparse

def parse() :
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', type=str, nargs='+')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse()
    _main_kv = dict()
    for filename in args.files :
        _key_val = regex_key_val.findall(open(filename).read())
        for k, v in _key_val :
            if k not in _main_kv :
                _main_kv[k] = v
            else :
                continue
        pass
    _main_kv = sorted(list(_main_kv.items()), key=lambda x : x[0])
    print('\n'.join([x+' '+y for x,y in _main_kv]))
