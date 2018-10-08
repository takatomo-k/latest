import json
import sys
import os
import argparse
import re
import math
from tqdm import tqdm
from pydub import AudioSegment 

def parse() :
    parser = argparse.ArgumentParser()
    parser.add_argument('--ali', type=str, help='alignment for wav.scp, produced by gentle aligner')
    parser.add_argument('--wav_scp', type=str, help='path to wav.scp')
    parser.add_argument('--ngram', type=int, default=1, help='number of n-gram (data augmentation)')
    parser.add_argument('--output', type=str, help='output path for all wav, text, wav_scp')
    return parser.parse_args()

if __name__ == '__main__' :
    args = parse()
    re_key_wav = re.compile('(^[^\s]+) (.+)$', re.MULTILINE)
    list_key_wav = re_key_wav.findall(open(args.wav_scp).read())
    output_path = args.output
    os.makedirs(output_path, exist_ok=True)
    dict_key_wav = dict(list_key_wav)
    ali = json.load(open(args.ali))
    assert len(list_key_wav) == len(ali)
    new_key = []
    new_text = []
    new_wav_scp = []
    ngram = args.ngram
    for ii in tqdm(list(range(len(ali)))) :
        key = ali[ii]['key']
        wav = dict_key_wav[key]
        audio = AudioSegment.from_wav(wav)
        list_segment = ali[ii]['words']
        for jj in range(0, len(list_segment)-ngram+1) :
            valid=True

            for kk in range(ngram) :
                if list_segment[jj+kk]['case'] != 'success' :
                    valid=False
            if not valid :
                continue
            else :
                segment_start = list_segment[jj]
                segment_end = list_segment[jj+kk]
                segment_text = ' '.join([x['word'] for x in list_segment[jj:jj+kk]])
            if valid :
                new_wav_name, ext = os.path.splitext(wav)
                new_wav_name = os.path.basename(new_wav_name)
                new_wav_ending = '_ng{}_w{}'.format(ngram, jj)
                new_wav_name += new_wav_ending
                new_wav_file = '{}{}'.format(new_wav_name, ext)
                new_wav_file = os.path.join(output_path, new_wav_file)

                start = math.floor(segment_start['start'] * 1000)
                end = math.ceil(segment_end['end'] * 1000)
                segmented_audio = audio[start:end]
                segmented_audio.export(new_wav_file)
                new_wav_scp.append(new_wav_file) 
                new_key.append(key+new_wav_ending)
                new_text.append(segment_start['word'])
            else :
                pass
            pass
        pass
    assert len(new_key) == len(new_wav_scp) == len(new_text)
    # write all #
    with open(os.path.join(output_path, 'wav.scp'), 'w') as f :
        for key, wav in zip(new_key, new_wav_scp) :
            f.write('{} {}\n'.format(key, wav))
    with open(os.path.join(output_path, 'text'), 'w') as f :
        for key, text in zip(new_key, new_text) :
            f.write('{} {}\n'.format(key, text))
    print('=== FINISH ===')
    pass
