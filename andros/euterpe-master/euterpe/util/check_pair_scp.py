import sys
import re

re_kv = re.compile('(^[^\s]+) (.+)$', re.MULTILINE)
if __name__ == '__main__' :
    files_x = sys.argv[1:]
    key_first = []
    with open(files_x[0]) as f :
        key_first = list(zip(*re_kv.findall(f.read())))[0]
        print('NUM OF UTTS: {}'.format(len(key_first)))

    for fii in files_x[1:] :
        with open(fii) as f :
            key_next = list(zip(*re_kv.findall(f.read())))[0]
            assert len(key_first) == len(key_next)
            for ii in range(len(key_first)) :
                assert key_first[ii] == key_next[ii], "[ERROR] key different in file {}, line {}".format(fii, ii) 
            print('[PASSED] {}'.format(fii))
    pass
