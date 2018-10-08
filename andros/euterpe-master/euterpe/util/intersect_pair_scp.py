import sys
from utilbox.regex_util import regex_key_val

if __name__ == '__main__':
    set_global = None
    for fii in sys.argv[1:] :
        with open(fii) as f :
            list_kv = regex_key_val.findall(f.read())
            if set_global is None :
                set_global = set([x[0] for x in list_kv])
            else:
                set_global.intersection_update([x[0] for x in list_kv])

    for fii in sys.argv[1:] :
        with open(fii) as f, open(fii+'.fix', 'w') as g :
            list_kv = regex_key_val.findall(f.read())

            for k,v in list_kv :
                if k in set_global :
                    g.write('{} {}\n'.format(k, v))
            pass
        pass
    pass
