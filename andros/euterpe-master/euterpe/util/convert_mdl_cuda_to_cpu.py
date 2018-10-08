import sys
import torch

if __name__ == '__main__' :
    mdl_path = sys.argv[1]
    params = torch.load(mdl_path)
    for k, v in list(params.items()) :
        params[k] = v.cpu()
        pass
    torch.save(params, mdl_path)
    print("===FINISH===")
    pass
