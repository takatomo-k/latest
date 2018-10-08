import os
import json
import yaml
import torch
import importlib
import re

class ModelSerializer(object) :

    @staticmethod
    def save_config(filename, config) :
        assert ('class' in config), 'config should contain class name'
        json.dump(config, open(filename, 'w'), indent=4)

    @staticmethod
    def load_config(filename, package=None) :
        if isinstance(filename, str) :
            config = yaml.load(open(filename))
        else :
            config = filename
        _fullname = re.findall("\'([A-Za-z0-9\.\_]+)\'", config['class'])[0]
        module_name, class_name = re.findall('^([A-Za-z0-9\.\_]+)\.([A-Za-z0-9\.\_]+)$', _fullname)[0]
        if package is not None :
            module_name = '{}.{}'.format(package, module_name)
        class_obj = getattr(importlib.import_module(module_name), class_name)
        del config['class']
        return class_obj(**config)

    @staticmethod
    def load_config_and_state(filename_cfg=None, filename_state=None, package=None) :
        assert (filename_cfg is not None) or (filename_state is not None)
        if filename_cfg is None :
            filename_cfg = os.path.splitext(filename_state)[0]+'.cfg'
        if filename_state is None :
            filename_state = os.path.splitext(filename_cfg)[0]+'.mdl'
        model = ModelSerializer.load_config(filename_cfg, package)
        model.load_state_dict(torch.load(filename_state))
        return model
    
    @staticmethod
    def convert_param_to_cpu(state_dict) :
        res = {}
        for k, v in list(state_dict.items()) :
            res[k] = v.cpu()
        return res
    pass

