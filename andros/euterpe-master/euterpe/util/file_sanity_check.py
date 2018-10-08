import json
import sys
import os
from utilbox.parse_util import all_config_load

def recursive_check(key_val) :
    for k, v in key_val.items() :
        if isinstance(v, dict) :
            recursive_check(v)
        elif isinstance(v, str) :
            if not os.path.exists(v) :
                print("file {} not exist !".format(v))
        elif v is None :
            pass
        else :
            raise Warning('wrong class object {}'.format(type(v)))

if __name__ == '__main__' :
    for fname in sys.argv[1:] :
        recursive_check(all_config_load(fname))
