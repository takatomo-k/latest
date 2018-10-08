import json
import sys
import os
from utilbox.regex_util import regex_key_val

if __name__ == '__main__':
    list_kv = regex_key_val.findall(sys.stdin.read())
    print(json.dumps(dict(list_kv), indent=2))
    pass
