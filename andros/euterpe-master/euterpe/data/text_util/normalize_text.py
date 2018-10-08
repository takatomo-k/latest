import argparse
from euterpe.data.text_util.text_util import sequence_to_text, text_to_sequence
from utilbox.regex_util import regex_key_val

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, help='text file')
    parser.add_argument('--cleaners', type=str, nargs='+', help='cleaner type (english_cleaners)', default=['english_cleaners'])
    args = parser.parse_args()
    with open(args.text) as f :
        all_lines = f.read()
        list_kv = regex_key_val.findall(all_lines)
        for k, v in list_kv :
            v_normalize = sequence_to_text(text_to_sequence(v, args.cleaners))
            print('{} {}'.format(k, v_normalize))
