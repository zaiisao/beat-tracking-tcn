"""
Ben Hayes 2020

ECS7006P Music Informatics

Coursework 1: Beat Tracking

File: beat_tracking_tcn/models/tcn.py
Description: Implements a non-causal temporal convolutional network based on
             the model used in Bai et al 2018 [1]
References:
[1] Bai, S., Kolter, J.Z. and Koltun, V., 2018. An empirical evaluation of
    generic convolutional and recurrent networks for sequence modeling. arXiv
    preprint arXiv:1803.01271.
"""
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class NonCausalTemporalLayer(nn.Module):
    """
    Implements a non-causal temporal block. Based off the model described in
    Bai et al 2018, but with the notable difference that the dilated temporal
    convolution is non-causal.

    Also implements a parallel residual convolution, allowing detail from the
    original signal to influence the output.
    """
    def __init__(
            self,
            inputs,
            outputs,
            dilation,
            kernel_size=5,
            stride=1,
            padding=4,
            dropout=0.1):
        """
        Construct an instance of NonCausalTemporalLayer

        Arguments:
            inputs {int} -- Input size in samples
            outputs {int} -- Output size in samples
            dilation {int} -- Size of dilation in samples

        Keyword Arguments:
            kernel_size {int} -- Size of convolution kernel (default: {5})
            stride {int} -- Size of convolution stride (default: {1})
            padding {int} -- How much padding to apply in total. Note, this is
                             halved and applied equally at each end of the
                             convolution to make the model non-causal.
                             (default: {4})
            dropout {float} -- The probability of dropping out a connection
                               during training. (default: {0.1})
        """
        super(NonCausalTemporalLayer, self).__init__()
        
        self.in_ch = inputs  #MJ= 16
        self.out_ch = outputs #MJ: = 16# JA: These properties are added so BeatFCOS can retrieve input/output channel size

        self.conv1 = nn.Conv1d(  #MJ: the input tensor: (B,C,L)
                inputs,
                outputs,
                kernel_size,
                #MJ: stride=stride, In TCN block, there are two conv1d. Only the second downsamples.
                stride= 1,
                padding=int(padding / 2),
                dilation=dilation)
        self.conv1 = weight_norm(self.conv1)
        self.elu1 = nn.ELU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
                inputs,
                outputs,
                kernel_size,
                stride=stride,
                padding=int(padding / 2),
                dilation=dilation)
        self.conv2 = weight_norm(self.conv2)
        self.elu2 = nn.ELU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(inputs, outputs, 1)\
            if inputs != outputs else None
        self.elu3 = nn.ELU()

        self._initialise_weights(self.conv1, self.conv2, self.downsample)

    def forward(self, x):
        """
        Feed a tensor forward through the layer.

        Arguments:
            x {torch.Tensor} -- A PyTorch tensor of size specified in the
                                constructor.

        Returns:
            torch.Tensor -- A PyTorch tensor of size specified in the
                            constructor.
        """
        y = self.conv1(x)   #MJ: the input tensor: (B,C,L) = (8,16,3000): =>  MJ, Runtime Error: F.conv1d(input, weight, bias, self.stride,=> Input=torch.cuda.FloatTensor, weight=torch.FloatTensor, mismatch)
        y = self.elu1(y)
        y = self.dropout1(y)
        y = self.conv2(y)
        y = self.elu2(y)
        y = self.dropout2(y)

        if self.downsample is not None:
            y = y + self.downsample(x)

        y = self.elu3(y)

        return y

    def _initialise_weights(self, *layers):
        for layer in layers:
            if layer is not None:
                layer.weight.data.normal_(0, 0.01)


class NonCausalTemporalConvolutionalNetwork(nn.Module):
    """
    Implements a non-causal temporal convolutional network. Based off the model
    described in Bai et al 2018. Initialises and forwards through a number of
    NonCausalTemporalLayer instances, depending on construction parameters.
    """
    def __init__(self, inputs, channels, kernel_size=5, dropout=0.1):
        """
        Construct a NonCausalTemporalConvolutionalNetwork.

        Arguments:
            inputs {int} -- Network input length: in_channels
            channels {list[int]} -- List containing number of channels each
                                    constituent temporal layer should have.

        Keyword Arguments:
            kernel_size {int} -- Size of dilated convolution kernels.
                                 (default: {5})
            dropout {float} -- The probability of dropping out a connection
                               during training. (default: {0.1})
        """
        super(NonCausalTemporalConvolutionalNetwork, self).__init__()

       
        
        #class ModuleList(Module):
       # Holds submodules in a list. torch.nn.ModuleList` can be indexed like a regular Python list, but
       #  modules it contains are properly registered, and will be visible by all torch.nn.Module` methods.

       # Args:    modules (iterable, optional): an iterable of modules to add

        #MJ: self.layers = []
        self.blocks = nn.ModuleList()
        
        n_levels = len(channels)  #MJ:  n_levels 11

        #for i in range(n_levels - 1): #i=0,,,(n_levels - 2)
        for i in range(n_levels): #i=0,,,(n_levels - 1)
            dilation = 2 ** i

            n_channels_in = channels[i - 1] if i > 0 else inputs
            n_channels_out = channels[i]

            self.blocks.append(
                NonCausalTemporalLayer( ##MJ: Conv1d: input tensor=(B,C,L)
                    n_channels_in,
                    n_channels_out,
                    dilation,
                    kernel_size,
                    stride=1, # JA: Because the stride is 1, the upper layers of TCN will have the temporal size as the lower layers
                    padding=(kernel_size - 1) * dilation,
                    dropout=dropout
                )
            )
        # # i = n_levels - 1
        # dilation = 2 ** (n_levels - 1)

        # n_channels_in = channels[n_levels - 2]
        # n_channels_out = channels[n_levels - 1]

        # self.blocks.append(
        #     NonCausalTemporalLayer( ##MJ: Conv1d: input tensor=(B,C,L)
        #         n_channels_in,
        #         n_channels_out,
        #         dilation,
        #         kernel_size,
        #         stride=2, # JA: Because the stride is 2, the upper layers of TCN will have the half  size of the lower layers
        #         padding=(kernel_size - 1) * dilation,
        #         dropout=dropout
        #     )
        # )
        
        #self.net = nn.Sequential(*self.layers) #self.layers is a list of module objects, not visible to nn.torch.Module methods

    def forward(self, x, number_of_backbone_layers=None, base_image_level_from_top=None):
        """
        Feed a tensor forward through the network.

        Arguments:
            x {torch.Tensor} -- A PyTorch tensor of size specified in the
                                constructor.

        Returns:
            torch.Tensor -- A PyTorch tensor of size determined by the final
                            temporal convolutional layer.
        """
        #y = self.net(x)
        #return y
        
         # get the shape of the layer just below the top; It will be used as the base resolution for the
         # FPN
        # base_level_image_shape = self.blocks[-base_image_level_from_top].shape
        
        #Execute the tcn and get the results of the top two blocks:
        
        last_count = 1 if number_of_backbone_layers is None else number_of_backbone_layers
        base_level_image_shape = None
        results = []
        start_level_for_fpn = len(self.blocks) - last_count
        
        for i, block in enumerate(self.blocks): # execute each TCN block
            x = block(x)
            if i >= start_level_for_fpn:  
                results.append(x)
        
            
            if (len(self.blocks) - 1) - i == base_image_level_from_top: #MJ: len=11: (11 -1) -9 == 1: the level 9 is used as the base image shape  
                base_level_image_shape = x.shape

        return results, base_level_image_shape