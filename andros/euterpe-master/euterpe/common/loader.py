import os
import json
import yaml
from enum import Enum
from utilbox.regex_util import regex_key_val, regex_key
from utilbox.parse_util import all_config_load

from ..data.data_iterator import DataIterator, TextIterator, DataIteratorNP

class LoaderDataType(Enum) :
    TYPE_H5 = 'h5'
    TYPE_NP = 'np'

def loader_seq2seq(data_config, data_type=LoaderDataType.TYPE_H5, driver=None, in_memory=False) :
    """
    data_type : 
        + h5 (using hdf5 file pytables)
        + npz (using npz file)
    """
    assert data_type in [LoaderDataType.TYPE_H5, LoaderDataType.TYPE_NP] 
    # 1. get data #
    if data_type == LoaderDataType.TYPE_H5 :
        train_feat_iterator = DataIterator(data_config['feat_train'], driver=driver)
    else :
        train_feat_iterator = DataIteratorNP(data_config['feat_train'], in_memory=in_memory)
    train_text_iterator = TextIterator(data_config['text_train'])     

    # 2. add blank '<b/eos>' #
    map_text2idx = json.load(open(data_config['vocab']))
    train_text_iterator.set_map_text2idx(map_text2idx)
    
    # 3. get dev & test data #
    if data_type == LoaderDataType.TYPE_H5 :
        dev_feat_iterator = DataIterator(data_config['feat_dev'], driver=driver)
    else :
        dev_feat_iterator = DataIteratorNP(data_config['feat_dev'], in_memory=in_memory)
    dev_text_iterator = TextIterator(data_config['text_dev'])
    dev_text_iterator.set_map_text2idx(map_text2idx)

    if data_type == LoaderDataType.TYPE_H5 :
        test_feat_iterator = DataIterator(data_config['feat_test'], driver=driver)
    else :
        test_feat_iterator = DataIteratorNP(data_config['feat_test'], in_memory=in_memory)
    test_text_iterator = TextIterator(data_config['text_test'])
    test_text_iterator.set_map_text2idx(map_text2idx)

    feat_iterator = {'train':train_feat_iterator, 'dev':dev_feat_iterator, 'test':test_feat_iterator}
    text_iterator = {'train':train_text_iterator, 'dev':dev_text_iterator, 'test':test_text_iterator}
    return feat_iterator, text_iterator

def loader_seq2seq_single(data_config, data_type=LoaderDataType.TYPE_H5, driver=None, in_memory=False) :

    # 1. get data #
    if data_type == LoaderDataType.TYPE_H5 :
        train_feat_iterator = DataIterator(data_config['feat_train'], driver=driver)
    else :
        train_feat_iterator = DataIteratorNP(data_config['feat_train'], in_memory=in_memory)
    train_text_iterator = TextIterator(data_config['text_train'])
    # 2. add blank '<b/eos>' #
    map_text2idx = json.load(open(data_config['vocab']))
    train_text_iterator.set_map_text2idx(map_text2idx)

    return train_feat_iterator, train_text_iterator

def loader_seq2seq_single_feat(data_config, data_type=LoaderDataType.TYPE_H5, driver=None, in_memory=False) :
    # 1. get feat #
    if data_type == LoaderDataType.TYPE_H5 :
        train_feat_iterator = DataIterator(data_config['feat_train'], driver=driver)
    else :
        train_feat_iterator = DataIteratorNP(data_config['feat_train'], in_memory=in_memory)
    return train_feat_iterator

def loader_seq2seq_single_text(data_config) :
    # 1. get text #
    train_text_iterator = TextIterator(data_config['text_train'])
    # 2. add blank '<b/eos>' #
    map_text2idx = json.load(open(data_config['vocab']))
    train_text_iterator.set_map_text2idx(map_text2idx)
    return train_text_iterator

############################
### LOADER WITH SET (V2) ###
### FOR EASY PARTION SET ###
############################
class DataLoader :
    @staticmethod
    def _subset_data(complete_list, partial_key) :
        complete_dict = dict(complete_list)
        new_list = []
        for key in partial_key :
            new_list.append((key, complete_dict[key]))
        return new_list

    @staticmethod
    def _read_key(path) :
        return regex_key.findall(open(path).read())

    @staticmethod
    def _read_key_val(path) :
        return regex_key_val.findall(open(path).read())

    @staticmethod
    def _check_intersect(list_iterator) :
        for ii in range(len(list_iterator)) :
            for jj in range(ii+1, len(list_iterator)) :
                assert not set(list_iterator[ii].key).intersection(set(list_iterator[jj].key))
        pass


    @staticmethod
    def load_feat_single(feat_path, key=None, data_type=LoaderDataType.TYPE_NP, in_memory=False) :
        assert data_type == LoaderDataType.TYPE_NP, "currently only support TYPE_NP"
        feat_kv = DataLoader._read_key_val(feat_path) 
        feat_len_all_path = '{}_len{}'.format(*os.path.splitext(feat_path))
        feat_len_kv = DataLoader._read_key_val(feat_len_all_path)

        if key is not None :
            if os.path.exists(key) :
                key = DataLoader._read_key(key)
            feat_kv = DataLoader._subset_data(feat_kv, key)
            feat_len_kv = DataLoader._subset_data(feat_len_kv, key)
        feat_iterator = DataIteratorNP(feat_kv=feat_kv, feat_len_kv=feat_len_kv,
                in_memory=in_memory)
        return feat_iterator


    @staticmethod
    def load_feat(data_config, data_type=LoaderDataType.TYPE_NP, in_memory=False) :
        assert data_type == LoaderDataType.TYPE_NP, "currently only support TYPE_NP"
        feat_all = DataLoader._read_key_val(data_config['feat']['all'])
        feat_len_all_path = '{}_len{}'.format(*os.path.splitext(data_config['feat']['all']))
        feat_len_all = DataLoader._read_key_val(feat_len_all_path)

        if 'set' in data_config['feat'] :
            feat_iterator = {}
            for set_name in ['train', 'dev', 'test'] :
                _key = DataLoader._read_key(data_config['feat']['set'][set_name])
                _feat_kv = DataLoader._subset_data(feat_all, _key)
                _feat_len_kv = DataLoader._subset_data(feat_len_all, _key)
                feat_iterator[set_name] = DataIteratorNP(
                        feat_kv=_feat_kv, feat_len_kv=_feat_len_kv, in_memory=in_memory)
            # check iterator set
            DataLoader._check_intersect(list(feat_iterator.values()))
            return feat_iterator
        else :
            raise ValueError()
        pass
    @staticmethod
    def load_text_single(text_path, key=None, vocab=None) :
        text_kv = DataLoader._read_key_val(text_path) 

        if key is not None :
            if os.path.exists(key) :
                key = DataLoader._read_key(key)
            text_kv = DataLoader._subset_data(text_kv, key)
        text_iterator = TextIterator(text_kv=text_kv)
        text_iterator.set_map_text2idx(vocab)
        return text_iterator

    @staticmethod
    def load_text(data_config) :
        text_all = DataLoader._read_key_val(data_config['text']['all'])
        if 'set' in data_config['text'] :
            text_iterator = {}
            _vocab = yaml.load(open(data_config['text']['vocab']))
            for set_name in ['train', 'dev', 'test'] :
                _key = DataLoader._read_key(data_config['text']['set'][set_name])
                _text_kv = DataLoader._subset_data(text_all, _key)
                text_iterator[set_name] = TextIterator(text_kv=_text_kv)
                text_iterator[set_name].set_map_text2idx(_vocab)
            # check iterator set
            DataLoader._check_intersect(list(text_iterator.values()))
            return text_iterator
        else :
            raise ValueError()
        pass
