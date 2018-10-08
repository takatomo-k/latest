from torch import nn
from torch.nn import Module
from torch.nn import functional as F
from ...nn.modules.conv import Conv2dEv

class ResidualFNN(Module) :
    def __init__(self, out_features, num_layers, fn_act=F.leaky_relu, use_bn=False, skip=1) :
        super(ResidualFNN, self).__init__()
        self.num_layers = num_layers
        self.linear_lyr = nn.ModuleList([nn.Linear(out_features, out_features) for _ in range(num_layers)])
        self.use_bn = use_bn
        if self.use_bn :
            self.bn_lyr = nn.ModuleList([nn.BatchNorm1d(out_features) for _ in range(num_layers)])
        self.fn_act = fn_act
        self.skip = skip
        
    def forward(self, x) :
        list_x = [x] # x_0 #
        for ii in range(1, self.num_layers+1) :
            x = self.linear_lyr[ii-1](x) # x_ii [1 .. N_LYR] #
            if self.use_bn :
                x = self.bn_lyr[ii-1](x)
            x = self.fn_act(x)
            if (ii)%self.skip == 0 :
                # apply residual connection #
                x = x + list_x[ii-self.skip]
            list_x.append(x)
        return list_x[-1]
        pass

class ResidualBlock2D(Module) :
    def __init__(self, out_channels, kernel_size=[3, 3], stride=1, 
            num_layers=2, use_bn=False, fn_act=F.leaky_relu) :
        super().__init__()
        self.out_channels = out_channels
        self.stride=stride

        self.num_layers = num_layers
        self.use_bn = use_bn
        self.fn_act = fn_act

        self.conv_lyrs = nn.ModuleList()
        if self.use_bn :
            self.bn_lyrs = nn.ModuleList()
        for ii in range(num_layers) :
            self.conv_lyrs.append(Conv2dEv(out_channels, out_channels, 
                kernel_size=kernel_size, stride=stride, padding='same'))
            if self.use_bn :
                self.bn_lyrs.append(nn.BatchNorm2d(out_channels))
        pass
    
    def forward(self, x) :
        transformed_x = x
        for ii in range(self.num_layers) :
            transformed_x = self.conv_lyrs[ii](transformed_x) 
            if ii == self.num_layers - 1 : # if ii is last layer #
                transformed_x = x + transformed_x # original x + projected x
            else :
                pass
            transformed_x = self.fn_act(transformed_x)

            # use bn after nonlinearity
            # ref : https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md
            if self.use_bn :
                transformed_x = self.bn_lyrs[ii](transformed_x)
        return transformed_x
