import os
import argparse
import subprocess

KALDI_COMPUTE_WER = '/home/is/andros-tj/rsch/asr/kaldi/src/bin/compute-wer'

def parse() :
    parser = argparse.ArgumentParser(description='python wrapper for Kaldi\'s computer-wer')
    parser.add_argument('--ref', required=True, help='reference transcription  (with key)')
    parser.add_argument('--hyp', required=True, help='hypothesis transcription (with key)')
    return parser.parse_args()
    pass

if __name__ == '__main__' :
    args = parse()
    print((subprocess.check_output('{} --text --mode=present ark:{} ark:{} '.format(
        os.path.join(KALDI_COMPUTE_WER), 
        os.path.abspath(args.ref), 
        os.path.abspath(args.hyp)), 
        shell=True)).decode('ascii'), 
        end="")
    pass
