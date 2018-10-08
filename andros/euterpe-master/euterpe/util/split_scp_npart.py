import os
import sys
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--scp', type=str, default=5)
parser.add_argument('--size', type=int, nargs='+')
parser.add_argument('--suffix', type=str, nargs='+', help='suffix for file-n')
parser.add_argument('--random', type=int, default=-1)
parser.add_argument('--use', type=(lambda x : x.lower() in ['t', 'true']), nargs='+')
parser.add_argument('--out', type=str, default=None, help='(optional) output path for the result file')
if __name__ == '__main__' :
    args = parser.parse_args()

    scp_path = args.scp
    size = args.size
    suffix = args.suffix
    assert len(size) == len(suffix)
    with open(scp_path) as f :
        lines = f.readlines()
    if len(lines) != sum(size) :
        print('warning : not all lines are used')

    order = list(range(len(lines)))
    if args.random > -1 :
        np.random.seed(args.random)
        np.random.shuffle(order)

    fname, ext = os.path.splitext(scp_path)
    start, end = 0, 0 
    for ii in range(len(size)) :
        start = 0 if ii == 0 else end
        end = start + size[ii] 
        if args.use is None or args.use[ii] :
            _new_filepath = fname+'_'+suffix[ii]+ext
            if args.out is not None :
                _new_filepath = os.path.join(args.out, os.path.basename(_new_filepath))
            with open(os.path.join(args.out, _new_filepath) if args.out is not None else _new_filepath, 'w') as f :
                f.writelines([lines[rr] for rr in sorted(order[start:end])])

    print('===FINISH===')
