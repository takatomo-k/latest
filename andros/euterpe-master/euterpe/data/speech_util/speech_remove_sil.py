import sys
import argparse
import os
import tqdm
from utilbox.regex_util import regex_key_val
from subprocess import call
import librosa

def parse() :
    args = argparse.ArgumentParser()
    args.add_argument('--list', type=str, help='scp file contain <key> <path to wav>')
    args.add_argument('--start', type=float, default=1.0, help='remove silence in the beginning of file if the volume < N %')
    args.add_argument('--end', type=float, default=0.25, help='remove silence in the beginning of file if the volume < N %')
    args.add_argument('--suffix', type=str, default='_nosil', help='suffix for output')
    return args.parse_args()

def sox_sil_cmd(input_file, output_file, start, end) :
    call('sox {} {} silence 1.0 1 {}% reverse silence 1 0.1 {} reverse'.format(input_file, output_file, start, end), shell=True)

if __name__ == '__main__':
    args = parse()
    with open(args.list) as f :
        list_kv = regex_key_val.findall(f.read())
        tobj = tqdm.tqdm(list_kv, ncols=60)
        for k, v in tobj :
            basename, ext = os.path.splitext(v)
            output_file = '{}_{}{}'.format(basename, args.suffix, ext)
            sox_sil_cmd(v, output_file, args.start, args.end)
            if librosa.get_duration(librosa.load(output_file)[0]) < 0.1 :
                print("filename {} is empty after silence removal".format(output_file), file=sys.stderr)
            else :
                tobj.write('{} {}'.format(k, output_file))
            pass
        pass
    pass
