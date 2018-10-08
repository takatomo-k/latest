import sys
import re

if __name__ == '__main__' :
    """
    python convert_char2word.py <hypothesis-text> 
    """
    re_key_val = re.compile('(^[^\s]+) (.+)$', re.MULTILINE)
    text_scp = sys.argv[1]
    key_text_list = re_key_val.findall(open(text_scp).read())
    for ii in range(len(key_text_list)) :
        merged_text = key_text_list[ii][1].replace(' ', '').replace('<spc>', ' ')
        print('{} {}'.format(key_text_list[ii][0], merged_text))
    pass
