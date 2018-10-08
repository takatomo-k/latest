import json
import yaml
import os

def args_file_or_json(arg) :
    if os.path.exists(arg) :
        return json.load(open(arg))
    else :
        return json.loads(arg)
    pass

def all_config_load(arg) :
    if os.path.exists(arg) :
        file_ext = os.path.splitext(arg)[1]
        if file_ext == '.json' :
            return json.load(open(arg))
        elif file_ext in ['.yaml', '.yml'] :
            return yaml.load(open(arg))
        else :
            raise NotImplementedError("file ext not supported")
    else :
        return yaml.load(arg) # json can be parsed by yaml #
