enc_emb_size=256
enc_emb_drop=0.25
enc_rnn_sizes=[256, 256]
enc_rnn_bi=[True,True]
enc_rnn_skip=[1,1]
enc_rnn_drop=[0.0,0.25]
enc_rnn_cfgs={"type":"lstm", "bi":True}
downsampling=[False, False, False],
dec_emb_size=256
dec_emb_drop=0.25#en:0.8 ja:0.25
dec_emb_tied_weight=True,
# tying weight from char/word embedding with softmax layer
dec_rnn_sizes=512
dec_rnn_drop =0.25#en:0.8 ja:0.25
dec_rnn_cfgs={"type":"lstm"}
dec_cfg={"type":"standard_decoder"},
att_cfg={"type":"mlp"},
lr=0.001
EOS=2
SOS=1
maxlen={'word':30,'char':100,'p_word':30,'p_char':100}