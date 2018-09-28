import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Prenet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Prenet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.layer = nn.Sequential(OrderedDict([
             ('fc1', nn.Linear(self.input_size, self.hidden_size)),
             ('relu1', nn.ReLU()),
             ('dropout1', nn.Dropout(0.5)),
             ('fc2', nn.Linear(self.hidden_size, self.output_size)),
             ('relu2', nn.ReLU()),
             ('dropout2', nn.Dropout(0.5)),
        ]))

    def forward(self, input_):

        out = self.layer(input_)

        return out

class CBHG(nn.Module):
    def __init__(self, hidden_size, projection_size = 128, K=16, num_gru_layers=2, max_pool_kernel_size=2, is_post=False):
        super(CBHG, self).__init__()
        self.hidden_size = hidden_size
        self.num_gru_layers = num_gru_layers
        self.projection_size = projection_size
        self.convbank_list = nn.ModuleList()
        self.convbank_list.append(nn.Conv1d(in_channels=projection_size,
                                                out_channels=hidden_size,
                                                kernel_size=1,
                                                padding=int(np.floor(1/2))))

        for i in range(2, K+1):
            self.convbank_list.append(nn.Conv1d(in_channels=hidden_size,
                                                out_channels=hidden_size,
                                                kernel_size=i,
                                                padding=int(np.floor(i/2))))

        self.batchnorm_list = nn.ModuleList()
        for i in range(1, K+1):
            self.batchnorm_list.append(nn.BatchNorm1d(hidden_size))

        convbank_outdim = hidden_size * K
        if is_post:
            self.conv_projection_1 = nn.Conv1d(in_channels=convbank_outdim,
                                             out_channels=hidden_size * 2,
                                             kernel_size=3,
                                             padding=int(np.floor(3/2)))
            self.conv_projection_2 = nn.Conv1d(in_channels=hidden_size * 2,
                                               out_channels=projection_size,
                                               kernel_size=3,
                                               padding=int(np.floor(3/2)))
            self.batchnorm_proj_1 = nn.BatchNorm1d(hidden_size * 2)

        else:
            self.conv_projection_1 = nn.Conv1d(in_channels=convbank_outdim,
                                             out_channels=hidden_size,
                                             kernel_size=3,
                                             padding=int(np.floor(3 / 2)))
            self.conv_projection_2 = nn.Conv1d(in_channels=hidden_size,
                                               out_channels=projection_size,
                                               kernel_size=3,
                                               padding=int(np.floor(3 / 2)))
            self.batchnorm_proj_1 = nn.BatchNorm1d(hidden_size)

        self.batchnorm_proj_2 = nn.BatchNorm1d(projection_size)


        self.max_pool = nn.MaxPool1d(max_pool_kernel_size, stride=1, padding=1)
        self.highway = Highwaynet(self.projection_size)
        self.gru = nn.GRU(self.projection_size, self.hidden_size, num_layers=2,
                          batch_first=True,
                          bidirectional=True)


    def _conv_fit_dim(self, x, kernel_size=3):
        if kernel_size % 2 == 0:
            return x[:,:,:-1]
        else:
            return x

    def forward(self, input_):
        input_ = input_.contiguous()
        batch_size = input_.size()[0]

        convbank_list = list()
        convbank_input = input_.transpose(1,2)

        # Convolution bank filters
        for k, (conv, batchnorm) in enumerate(zip(self.convbank_list, self.batchnorm_list)):
            convbank_input = F.relu(batchnorm(self._conv_fit_dim(conv(convbank_input), k+1).contiguous()))
            convbank_list.append(convbank_input)

        # Concatenate all features
        conv_cat = torch.cat(convbank_list, dim=1)

        # Max pooling
        conv_cat = self.max_pool(conv_cat)[:,:,:-1]

        # Projection
        conv_projection = F.relu(self.batchnorm_proj_1(self._conv_fit_dim(self.conv_projection_1(conv_cat))))
        conv_projection = self.batchnorm_proj_2(self._conv_fit_dim(self.conv_projection_2(conv_projection)))

        #Residual
        conv_projection =conv_projection.transpose(1,2) + input_

        # Highway networks
        highway = self.highway.forward(conv_projection)

        #Bidirectional GRU
        self.gru.flatten_parameters()
        out, _ = self.gru(highway)

        return out


class Highwaynet(nn.Module):
    def __init__(self, num_units, num_layers=4):
        super(Highwaynet, self).__init__()
        self.num_units = num_units
        self.num_layers = num_layers
        self.gates = nn.ModuleList()
        self.linears = nn.ModuleList()
        for _ in range(self.num_layers):
            self.linears.append(nn.Linear(num_units, num_units))
            self.gates.append(nn.Linear(num_units, num_units))

    def forward(self, input_):

        out = input_

        # highway gated function
        for fc1, fc2 in zip(self.linears, self.gates):

            h = F.relu(fc1.forward(out))
            t = F.sigmoid(fc2.forward(out))

            c = 1. - t
            out = h * t + out * c

        return out

class AttentionDecoder(nn.Module):
    """
    Decoder with attention mechanism (Vinyals et al.)
    """
    def __init__(self, num_units):
        """
        :param num_units: dimension of hidden units
        """
        super(AttentionDecoder, self).__init__()
        self.num_units = num_units

        self.v = nn.Linear(num_units, 1, bias=False)
        self.W1 = nn.Linear(num_units, num_units, bias=False)
        self.W2 = nn.Linear(num_units, num_units, bias=False)

        self.attn_grucell = nn.GRUCell(num_units // 2, num_units)
        self.gru1 = nn.GRUCell(num_units, num_units)
        self.gru2 = nn.GRUCell(num_units, num_units)

        self.attn_projection = nn.Linear(num_units * 2, num_units)
        self.out = nn.Linear(num_units, hp.num_mels * hp.outputs_per_step)

    def forward(self, decoder_input, memory, attn_hidden, gru1_hidden, gru2_hidden):

        memory_len = memory.size()[1]
        batch_size = memory.size()[0]

        # Get keys
        keys = self.W1(memory.contiguous().view(-1, self.num_units))
        keys = keys.view(-1, memory_len, self.num_units)

        # Get hidden state (query) passed through GRUcell
        d_t = self.attn_grucell(decoder_input, attn_hidden)

        # Duplicate query with same dimension of keys for matrix operation (Speed up)
        d_t_duplicate = self.W2(d_t).unsqueeze(1).expand_as(memory)

        # Calculate attention score and get attention weights
        attn_weights = self.v(F.tanh(keys + d_t_duplicate).view(-1, self.num_units)).view(-1, memory_len, 1)
        attn_weights = attn_weights.squeeze(2)
        attn_weights = F.softmax(attn_weights)

        # Concatenate with original query
        d_t_prime = torch.bmm(attn_weights.view([batch_size,1,-1]), memory).squeeze(1)

        # Residual GRU
        gru1_input = self.attn_projection(torch.cat([d_t, d_t_prime], 1))
        gru1_hidden = self.gru1(gru1_input, gru1_hidden)
        gru2_input = gru1_input + gru1_hidden

        gru2_hidden = self.gru2(gru2_input, gru2_hidden)
        bf_out = gru2_input + gru2_hidden

        # Output
        output = self.out(bf_out).view(-1, hp.num_mels, hp.outputs_per_step)

        return output, d_t, gru1_hidden, gru2_hidden

    def inithidden(self, batch_size):
        if use_cuda:
            attn_hidden = Variable(torch.zeros(batch_size, self.num_units), requires_grad=False).cuda()
            gru1_hidden = Variable(torch.zeros(batch_size, self.num_units), requires_grad=False).cuda()
            gru2_hidden = Variable(torch.zeros(batch_size, self.num_units), requires_grad=False).cuda()
        else:
            attn_hidden = Variable(torch.zeros(batch_size, self.num_units), requires_grad=False)
            gru1_hidden = Variable(torch.zeros(batch_size, self.num_units), requires_grad=False)
            gru2_hidden = Variable(torch.zeros(batch_size, self.num_units), requires_grad=False)

        return attn_hidden, gru1_hidden, gru2_hidden
