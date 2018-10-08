import numpy as np
import tables
import re
import itertools
import os
import yaml

from tqdm import trange

from euterpe.config import constant
from utilbox.regex_util import regex_key_val

class DataIterator :
    """
    wrapper for speech features corpus
    """
    def __init__(self, path, driver=None) :
        if driver is None :
            self.storage = tables.open_file(path)
        else :
            self.storage = tables.open_file(path, driver=driver)
        self.feat = self.storage.root.feat
        self.key = np.array([x.decode('utf-8') if isinstance(x, bytes) else x for x in self.storage.root.key])
        self.pos = self.storage.root.pos

        # additional feature #
        self.idx2key = dict([(x, y) for x, y in enumerate(self.key[:])])
        self.key2idx = dict([(y, x) for x, y in enumerate(self.key[:])])
        assert len(self.idx2key) == len(self.key2idx)
        pass

    def get_feat_by_index(self, idxs) :
        if not isinstance(idxs, list) :
            return self.feat[slice(*self.pos[idxs])]

        _seqs = []
        for ii in idxs :
            _seqs.append(self.get_feat_by_index(ii))
        return _seqs
    
    def get_feat_length_by_key(self, keys) :
        if not isinstance(keys, list) :
            idx = self.key2idx[keys]
            return (self.pos[idx][1] - self.pos[idx][0])
        else :
            return [self.get_feat_length_by_key(x) for x in keys]
        pass

    def get_feat_length_by_index(self, idxs) :
        if not isinstance(idxs, list) :
            return self.pos[idxs][1] - self.pos[idxs][0]
        else :
            return [self.get_feat_length_by_index(x) for x in idxs]
        pass

    def get_feat_length(self) :
        return [int(x[1]-x[0]) for x in self.pos]

    def get_feat_dim(self) :
        return int(self.feat.shape[-1])

    def get_index_by_key(self, keys) :
        if not isinstance(keys, list) :
            return self.key2idx[keys]
        else :
            return [self.get_index_by_key(x) for x in keys]
        pass

    def get_key_by_index(self, idxs) :
        if not isinstance(idxs, list) :
            return self.key[idxs]
        else :
            return [self.get_key_by_index(idx) for idx in idxs]
    
    def get_feat_by_key(self, keys) :
        return self.get_feat_by_index(self.get_index_by_key(keys)) 

    def get_feat_stat(self, axis=0) :
        if not hasattr(self, '_feat_stat') or self._feat_stat is None :
            self._feat_stat = {'max':self.feat[:].max(axis=axis), 
                'min':self.feat[:].min(axis=axis), 
                'mean':self.feat[:].mean(axis=axis), 
                'std':self.feat[:].std(axis=axis)}
        return self._feat_stat
    pass

class DataIteratorNP:
    """
    wrapper for speech features (single feat iterator)
    """
    def __init__(self, 
            feat_path=None, feat_len_path=None, 
            feat_kv=None, feat_len_kv=None, 
            in_memory=False) :
        """
        feat_path : feat.scp (pair between key and feat path)
        feat_len_path : feat_len.scp (file pair between key and length)
        """
        self.in_memory=in_memory
        if feat_kv is not None :
            # case 1 : path & len_path is dict
            list_kv = feat_kv
            list_klen = feat_len_kv
        elif feat_path is not None :
            assert os.path.exists(feat_path), "feat.scp is not exist"
            if feat_len_path is None :
                # automatically len_path
                feat_len_path = '{}_len{}'.format(*os.path.splitext(feat_path))
            assert os.path.exists(feat_len_path), "feat_len.scp is not exist"
            self.feat_len_path = feat_len_path
            # read feat.scp
            list_kv = regex_key_val.findall(open(feat_path).read())
            list_klen = regex_key_val.findall(open(feat_len_path).read())
        else :
            raise ValueError()
        # create map
        self.key, self.feat_path = zip(*list_kv)
        _tmp_key, self.feat_len = zip(*list_klen)
        self.feat_len = [int(x) for x in self.feat_len]
        assert self.key == _tmp_key, "feat.scp key != feat_len.scp key"
        self.idx2key = dict([(x, y) for x, y in enumerate(self.key)])
        self.key2idx = dict([(y, x) for x, y in enumerate(self.key)])
        assert len(self.idx2key) == len(self.key2idx)
        if self.in_memory :
            self._load_feat_to_memory()
        pass

    def _load_feat_to_memory(self) :
        self._feats = []
        for ii in range(len(self.key2idx)) :
            self._feats.append(np.load(self.feat_path[ii]))

    def get_feat_by_index(self, idxs) :
        if not isinstance(idxs, list) :
            if self.in_memory :
                return self._feats[idxs]['feat']
            else :
                return np.load(self.feat_path[idxs])['feat']
        _seqs = []
        for ii in idxs :
            _seqs.append(self.get_feat_by_index(ii))
        return _seqs
    
    def get_feat_length_by_key(self, keys) :
        return self.get_feat_length_by_index(self.get_index_by_key(keys))

    def get_feat_length_by_index(self, idxs) :
        if not isinstance(idxs, list) :
            return self.feat_len[idxs]
        else :
            return [self.get_feat_length_by_index(x) for x in idxs]
        pass

    def get_feat_length(self) :
        return self.feat_len

    def get_feat_dim(self) :
        return int(self.get_feat_by_index(0).shape[-1])

    def get_index_by_key(self, keys) :
        if not isinstance(keys, list) :
            return self.key2idx[keys]
        else :
            return [self.get_index_by_key(x) for x in keys]
        pass

    def get_key(self) :
        return self.key

    def get_key_by_index(self, idxs) :
        if not isinstance(idxs, list) :
            return self.key[idxs]
        else :
            return [self.get_key_by_index(idx) for idx in idxs]
    
    def get_feat_by_key(self, keys) :
        return self.get_feat_by_index(self.get_index_by_key(keys)) 

    pass

class TextIterator :
    def __init__(self, path=None, text_kv=None, map_text2idx=None) :
        self.key = []
        self.text = []
        self.map_text2idx = None
        if path is not None :
            with open(path) as f :
                all_texts = regex_key_val.findall(f.read())
                _key, _text = list(zip(*all_texts))
                self.key, self.text = list(_key), list(_text)
                for ii in range(len(self.text)) :
                    self.text[ii] = self.text[ii].split()
        else :
            assert text_kv is not None and isinstance(text_kv, list)
            for kv in text_kv :
                self.key.append(kv[0])
                self.text.append(kv[1].split())

        # additional feature #
        self.idx2key = dict([(x, y) for x, y in enumerate(self.key[:])])
        self.key2idx = dict([(y, x) for x, y in enumerate(self.key[:])])
        assert len(self.idx2key) == len(self.key2idx)

        # set map_text2idx if provided #
        if map_text2idx is not None :
            if isinstance(map_text2idx, str) :
                map_text2idx = yaml.load(open(map_text2idx))
                pass
            self.set_map_text2idx(map_text2idx)
        pass
    
    def get_map_text2idx(self) :
        if self.map_text2idx is None : 
            # generate default map_text2idx #
            uset = set()
            for _text in self.text :
                uset.update(_text)
            uset = sorted(list(uset))
            return dict(list(zip(uset, list(range(len(uset))))))
        else :
            return self.map_text2idx

    def get_key_by_index(self, idxs) :
        if not isinstance(idxs, list) :
            return self.key[idxs]
        else :
            return [self.get_key_by_index(idx) for idx in idxs]

    def get_index_by_key(self, keys) :
        if not isinstance(keys, list) :
            return self.key2idx[keys]
        else :
            return [self.get_index_by_key(key) for key in keys]

    def set_map_text2idx(self, value) :
        self.map_text2idx = value

    def get_text_by_key(self, keys, convert_to_idx=True) :
        return self.get_text_by_index(self.get_index_by_key(keys), convert_to_idx=convert_to_idx)

    def get_text_by_index(self, idxs, convert_to_idx=True) :
        if not isinstance(idxs, list) :
            if convert_to_idx :
                _result = []
                for x in self.text[idxs] :
                    if x not in self.map_text2idx :
                        _result.append(constant.UNK)
                    else :
                        _result.append(self.map_text2idx[x])
                return _result
            else :
                return self.text[idxs]
        else :
            _texts = []
            for idx in idxs :
                _texts.append(self.get_text_by_index(idx, convert_to_idx))
            return _texts

    def get_text_length(self) :
        return [len(x) for x in self.get_text_by_index(list(range(len(self.text))))]
    pass
