import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

# sns.set(style="white")

"""
# OLD CODE
# TODO: REMOVE
def draw_softmax_alignment(att_mat, label, path='visualize_alignment.png', transpose=False) :
    if transpose :
        att_mat = att_mat.T
    d = pd.DataFrame(data=att_mat)
    f, ax = plt.subplots(figsize=(60, 30))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(d, cmap=cmap, vmax=att_mat.max(), vmin=0.0,
                square=True, xticklabels=5, yticklabels=10,
                linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)

    # ax.set_yticklabels(label, rotation='horizontal')
    ax.set_yticklabels(ax.yaxis.get_majorticklabels(), rotation=45)
    f.savefig(path, bbox_inches='tight', dpi=100)
    pass
"""

def crop_attention_matrix(attention_matrix, dec_len, enc_len) :
    assert len(attention_matrix.shape) == 3, "shape must be (batch x dec_len x enc len)"
    batch = attention_matrix.shape[0]
    cropped_atts = []
    for ii in range(batch) :
        cropped_atts.append(attention_matrix[ii, 0:dec_len[ii], 0:enc_len[ii]])
    return cropped_atts

def plot_softmax_attention(att_mat, label, path='plot_att.png', transpose=False) :
    if transpose :
        att_mat = att_mat.T

    fig, ax = plt.subplots()
    im = ax.imshow(
        att_mat,
        aspect='auto',
        origin='lower',
        interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Encoder timestep'
    plt.xlabel(xlabel)
    plt.ylabel('Decoder timestep')
    plt.tight_layout()
    plt.gca().invert_yaxis()
    plt.savefig(path, format='png')
    plt.close()
