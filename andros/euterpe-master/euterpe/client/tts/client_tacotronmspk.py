
from __init__ import *

import os
import io
import shutil
import pandas as pd
import yaml
from flask import Flask
from flask import request, jsonify, abort, render_template
import subprocess
from common.loader import DataLoader
import wave
import base64

app = Flask('client_tacotronmspk', instance_relative_config=True)
app.config.from_pyfile('config.cfg')

# pre-configuration #
data_cfg = yaml.load(open(app.config['DATA_CFG']))
# key2spk = yaml.load(open(data_cfg['misc']['key2spk']))
# key2feat = DataLoader._read_key_val(data_cfg['feat']['all'])
# key2feat = DataLoader._subset_data(key2feat, DataLoader._read_key(data_cfg['feat']['set']['train']))
# list_spk_id = list(set([key2spk[x] for x,_ in key2feat]))
list_spk_id=["A05F0453", "S00M0441", "S04F1261", "A03M0175"]
list_vocab = sorted(list(yaml.load(open(data_cfg['text']['vocab'])).keys()))
list_example=["d oh z o y o r o sh i k u o n e g a i sh i m a s u", 
        "n a n a j uh g o n e N t e i i m a s u k a r a", 
        "k o r e o m a m a ch i n i d e k a k e t e",
        "w a t a sh i d o m o g a o k o n a q t a s e N k oh k e N ky uh n a N d e s u k e r e d o m o"]
# model = "expr_tts_multi_v2/csj_feattaco-tacotron_bern_end-baseline_attmlp-spk_tanh-declastgroup-mixup_none-loss_l1-loss_freq_topn0.25_c0.5-lr2.5e-4-cut1000/best_model.mdl"
model = "expr_tts_multi_v2/csj_feattaco-tacotron_bern_end-baseline_attmlphistory-spk_tanh-declastgroup-mixup_none-loss_l1-loss_freq_topn0.25_c0.5-lr2.5e-4-cut1000/best_model.mdl"
#####################

def cmd_synth(text, spk_id) :
    _cmd_line = 'cd {base} && python script_tts/synth_tacotron_ftaco_v2.py \
            --model {model} \
            --feat_cfg cfg_feat/feat/taco_mel_f80.json --feat_second_cfg cfg_feat/feat/taco_raw_fft2048.json \
            --data_cfg cfg_dataset/dset_csj/first40000spk_np_taco_melf80.yaml  \
            --data_second_cfg cfg_dataset/dset_csj/first40000spk_np_taco_raw.yaml \
            --text "{text}" \
            --spk "{spk_id}"'.format(base=app.config['BASE'], model=model, text=text, spk_id=spk_id)
    result = subprocess.check_output(_cmd_line, shell=True)
    result = yaml.load(result)[0]['wav_post']
    # copy from abs to local $
    data_wav = open(result, 'rb')
    data_wav.seek(0)
    return base64.encodestring(data_wav.read()).decode('utf-8')

def text_validation(text) :
    _set_vocab = set(list_vocab)
    for ii in text.strip().split() :
        if ii not in _set_vocab :
            return False
    return True

@app.route('/synth', methods=['POST'])
def synth() :
    _data = request.form
    text = _data['text']
    spk_id = _data['spk_id']
    if not text_validation(text) :
        return render_template('index.html', list_spk_id=list_spk_id, list_vocab=list_vocab, list_example=list_example, synth_wav=None, error='word/vocab is out-of-bound')
    try :
        result = cmd_synth(text, spk_id)
    except :
        return render_template('index.html', list_spk_id=list_spk_id, list_vocab=list_vocab, list_example=list_example, synth_wav=None, error='backend has encountered some errors')
    return render_template('index.html', list_spk_id=list_spk_id, list_vocab=list_vocab, list_example=list_example, synth_wav=result, error=None)

@app.route('/')
def main() :
    return render_template('index.html', list_spk_id=list_spk_id, list_vocab=list_vocab, list_example=list_example, synth_wav=None, error=None)
