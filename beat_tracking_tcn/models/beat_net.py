"""
Ben Hayes 2020

ECS7006P Music Informatics

Coursework 1: Beat Tracking

File: beat_tracking_tcn/models/beat_net.py
Description: A CNN including a Temporal Convolutional Layer designed to predict
             a vector of beat activations from an input spectrogram (tested
             with mel spectrograms).
"""
import torch.nn as nn

from beat_tracking_tcn.models.tcn import NonCausalTemporalConvolutionalNetwork


class BeatNet(nn.Module):
    """
    PyTorch implementation of a BeatNet CNN. The network takes a
    mel-spectrogram input. It then learns an intermediate convolutional
    representation, and finally applies a non-causal Temporal Convolutional
    Network to predict a beat activation vector.

    The structure of this network is based on the model proposed in Davies &
    Bock 2019.
    """

    def __init__(
            self,
            input=(3000, 81),
            output=3000,
            channels=16,
            tcn_kernel_size=5,
            dropout=0.1,
            downbeats=False):
        """
        Construct an instance of BeatNet.

        Keyword Arguments:
            input {tuple} -- Input dimensions (default: {(3000, 81)}) = 2D image 
        # MJ: parser.add_argument("-b", "--batch_size",  type=int,    default=1,
        # help="Batch size to use for training.")
            output {int} -- Output dimensions (default: {3000})
            channels {int} -- Convolution channels (default: {16})
            tcn_kernel_size {int} -- Size of dilated convolution kernels.
                                     (default: {5})
            dropout {float} -- Network connection dropout probability.
                               (default: {0.1})
        """
        super(BeatNet, self).__init__()

        #MJ: orch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        self.conv1 = nn.Conv2d(1, channels, (3, 3), padding=(1, 0))  # 16 filters of size 3 × 3
                                                # The spectrogram is considered to be a 2D one channel tensor
        self.elu1 = nn.ELU()
        self.dropout1 = nn.Dropout(dropout)
        self.pool1 = nn.MaxPool2d((1, 3))  #MJ: kernel=(1,3): max pooling over 3 bins in the frequency direction; 

        self.conv2 = nn.Conv2d(channels, channels, (3, 3), padding=(1, 0))
        self.elu2 = nn.ELU()
        self.dropout2 = nn.Dropout(dropout)
        self.pool2 = nn.MaxPool2d((1, 3))

        self.conv3 = nn.Conv2d(channels, channels, (1, 8))  #MJ: kernel=(1,8): 16 filters of size 1 × 8 without pooling
        self.elu3 = nn.ELU()
        self.dropout3 = nn.Dropout(dropout)  #MJ: not used

    #MJ: conv1d():  input tensor=(B,C,H,W), L=HxW
    #MJ: https://stackoverflow.com/questions/54542682/how-to-properly-pass-2d-array-to-conv1d    
    #https://stackoverflow.com/questions/65790139/why-does-nn-conv1d-work-on-2d-feature-b-c-h-w 
    #https://stats.stackexchange.com/questions/295397/what-is-the-difference-between-conv1d-and-conv2d
                                                          
    #    MJ:Internally, this op reshapes the input tensors and invokes tf.nn.conv2d.
    #    For example, if data_format does not start with "NC", a tensor of shape [batch, in_width, in_channels]
    #    is reshaped to [batch, 1, in_width, in_channels], and the filter is reshaped to 
    #    [1, filter_width, in_channels, out_channels]. 
    #    The result is then reshaped back to [batch, out_width, out_channels] 
    #    (where out_width is a function of the stride and padding as in conv2d) and returned to the caller.
    
    # In summary, In 1D CNN, kernel moves in 1 direction. Input and output data of 1D CNN is 2 dimensional. 
    # Mostly used on Time-Series data.
    # When using Conv1d(), we have to keep in mind that we are most likely going to work with 2-dimensional inputs 
    # such as one-hot-encode DNA sequences or black and white pictures.
    #the height of your input data becomes the “depth” (or in_channels) the height of your input data becomes the “depth” (or in_channels),
    # and our rows become the kernel size. 
    
    #MJ: https://towardsdatascience.com/pytorch-basics-how-to-train-your-neural-net-intro-to-cnn-26a14c2ea29

    # In 2D CNN, kernel moves in 2 directions. Input and output data of 2D CNN is 3 dimensional. 
    # Mostly used on Image data.
    
        self.tcn = NonCausalTemporalConvolutionalNetwork( 
                                                          
            channels,
            [channels] * 11,
            tcn_kernel_size, #MJ: =5
            dropout)

        self.out = nn.Conv1d(16, 1 if not downbeats else 2, 1)  #MJ 16 = channels
        self.sigmoid = nn.Sigmoid()

    def forward(self, x): #MJ: https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460 : __call__ is already defined in nn.Module, will register all hooks and call your forward. That’s also the reason to call the module directly (output = model(data)) instead of model.forward(data).
        """
        Feed a tensor forward through the BeatNet.

        Arguments:
            x {torch.Tensor} -- A PyTorch tensor of size specified in the
                                constructor.

        Returns:
            torch.Tensor -- A PyTorch tensor of size specified in the
                            constructor.
        """
        #MJ: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        y = self.conv1(x)  #MJ: x = (B, C, H,W) = (B, 1, 3000, 81)
        y = self.elu1(y)
        y = self.dropout1(y)
        y = self.pool1(y)

        y = self.conv2(y)
        y = self.elu2(y)
        y = self.dropout2(y)
        y = self.pool2(y)

        y = self.conv3(y)
        y = self.elu3(y)  #MJ: The shape of y = # y.shape: [1, 16, 3000, 1]

        y = y.view(-1, y.shape[1], y.shape[2])  # y.shape: [1, 16, 3000, 1] => (1,16,3000)
        
        #MJ: In this way, small (overlapping) spectrogram snippets with a context of 5 frames get reduced 
        # to a single frame and 16 features, 16-dim feature vector, which retains the temporal resolution.
        
        #MJ: : The principal means by which the TCN is able to capture sequential structure is by learning filters via dilated convolutions.
        # By working on a highly sub-sampled feature representation compared to the raw audio, 
        # we can obtain a large temporal receptive field with far fewer layers and weights than the raw audio domain equivalent.
        
        # for any given temporal frame of the input, the (dilated) convolutions extend both forwards and backwards
        # in time. For purely causal operation the dilated convolutions are only performed using past data
        # up to the current temporal frame with no access to future information in the signal. 
        # In the context of real-time beat tracking, such causal processing would be essential 
        # (as well as the need to adapt many other components of the beat tracking system), 
        # but for this paper where all other processing steps are performed offline
        
        y = self.tcn(y) # y.shape: [1, 16, 3000] Which is the 1D tensor with channel of 16 and length of 3000 and can now enter the 1D TCN
        #  # Output of TCN y has the same shape as the input
        
        #MJ: so far, we had the backbone of the net
        #MJ: the following two layers constitute the output layer (head)
        #  self.out = nn.Conv1d(16, 1 if not downbeats else 2, 1)  #MJ 16 = channels
        #  self.sigmoid = nn.Sigmoid()
        
        y = self.out(y)
        y = self.sigmoid(y)

        return y
