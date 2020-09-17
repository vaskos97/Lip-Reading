import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
from collections import OrderedDict
from loss_functions import AngularPenaltySMLoss
from torch.nn.modules.utils import _triple

'''class ConvAngularPen(nn.Module):
    def __init__(self, num_classes=300, loss_type='arcface'):
        super(ConvAngularPen, self).__init__()
        #self.convlayers = ConvNet()
        self.criterion = AngularPenaltySMLoss(512, num_classes, loss_type=loss_type)'''

class NLLSequenceLoss(nn.Module):
    """
    Custom loss function.
    Returns a loss that is the sum of all losses at each time step.
    """
    def __init__(self):
        super(NLLSequenceLoss, self).__init__()
        self.criterion = nn.NLLLoss(reduction='none')


    def forward(self, input, length, target):
        loss = []
        transposed = input.transpose(0, 1).contiguous()
        for i in range(transposed.size(0)):
            loss.append(self.criterion(transposed[i,], target.squeeze(0)).unsqueeze(1))
        loss = torch.cat(loss, 1)
        #print('loss:',loss)
        mask = torch.zeros(loss.size(0), loss.size(1)).float().cuda()

        for i in range(length.size(0)):
            L = min(mask.size(1), length[i])
            mask[i,L-1] = 1.0
        #print('mask:',mask)
        #print('mask * loss',mask*loss)
        loss = (loss * mask).sum() / mask.sum()
        return loss
        
        #return loss/transposed.size(0)

def _validate(modelOutput, length, labels, total=None, wrong=None):

    averageEnergies = torch.sum(modelOutput.data, 1)
    for i in range(modelOutput.size(0)):
        #print(modelOutput[i,:length[i]].sum(0).shape)
        averageEnergies[i] = modelOutput[i,:length[i]].sum(0)

    maxvalues, maxindices = torch.max(averageEnergies, 1)

    count = 0

    for i in range(0, labels.squeeze(1).size(0)):
        l = int(labels.squeeze(1)[i].cpu())
        if total is not None:
            if l not in total:
                total[l] = 1
            else:
                total[l] += 1 
        if maxindices[i] == labels.squeeze(1)[i]:
            count += 1
        else:
            if wrong is not None:
               if l not in wrong:
                   wrong[l] = 1
               else:
                   wrong[l] += 1

    return (averageEnergies, count)


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=(1,2,2), stride=(1,2,2)))


class Dense3D(torch.nn.Module):
    def __init__(self, options, growth_rate=32, num_init_features=64, bn_size=4, drop_rate=0):
        super(Dense3D, self).__init__()
        #block_config = (6, 12, 24, 16)
        block_config = (4, 8, 12, 8)

        self.features = nn.Sequential(OrderedDict([
            #('conv0', nn.Conv3d(3, num_init_features, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False)),
            ('conv0', SpatioTemporalConv(3, num_init_features, kernel_size=(5, 7, 7), padding=(2, 3, 3), stride=(1, 2, 2), bias=False)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))),
        ]))
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):

            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)

            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm

        self.features.add_module('norm5', nn.BatchNorm3d(num_features))
        self.gru1 = nn.GRU(536*3*3,256, bidirectional=True, batch_first=True)
        self.gru2 = nn.GRU(512, 256,  bidirectional=True, batch_first=True)
        #self.gru3 = nn.GRU(512, 256, bidirectional=True, batch_first=True)


        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 300)
            
        )
        self.loss = NLLSequenceLoss
        #self.loss = ConvAngularPen


    def validator_function(self):
        return _validate
    

    def forward(self, x):
        self.gru1.flatten_parameters()
        self.gru2.flatten_parameters()
        #self.gru3.flatten_parameters()
        f2 = self.features(x)
#        print(x.size())
        f2 = f2.permute(0, 2, 1, 3, 4).contiguous()
        B, T, C ,H ,W = f2.size()
        f2 = f2.view(B, T, -1)
        f2, _ = self.gru1(f2)
        f2, _ = self.gru2(f2)
        f2 = self.fc(f2).log_softmax(-1)
        return f2
    
class Swish(nn.Module):

    def __init__(self, inplace=False):
        super().__init__()

        self.inplace = True

    def forward(self, x):
        if self.inplace:
            x.mul_(F.sigmoid(x))
            return x
        else:
            return x * F.sigmoid(x)
        
        
class Mish(nn.Module):
    
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.tanh(F.softplus(input))
    
class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
    
class DropBlock2D(nn.Module):
    r"""Randomly zeroes spatial blocks of the input tensor.
    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.
    Args:
        keep_prob (float, optional): probability of an element to be kept.
        Authors recommend to linearly decrease this value from 1 to desired
        value.
        block_size (int, optional): size of the block. Block size in paper
        usually equals last feature map dimensions.
    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)
    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890
    """

    def __init__(self, keep_prob=0.9, block_size=7):
        super(DropBlock2D, self).__init__()
        self.keep_prob = keep_prob
        self.block_size = block_size

    def forward(self, input):
        if not self.training or self.keep_prob == 1:
            return input
        gamma = (1. - self.keep_prob) / self.block_size ** 2
        for sh in input.shape[3:]:
            gamma *= sh / (sh - self.block_size + 1)
        M = torch.bernoulli(torch.ones_like(input) * gamma)
        Msum = F.conv2d(M,
                        torch.ones((input.shape[2], 1, self.block_size, self.block_size)).to(device=input.device,
                                                                                             dtype=input.dtype),
                        padding=self.block_size // 2,
                        groups=input.shape[2])
        torch.set_printoptions(threshold=5000)
        mask = (Msum < 1).to(device=input.device, dtype=input.dtype)
        return input * mask * mask.numel() /mask.sum() #TODO input * mask * self.keep_prob ?
    
    
class SpatioTemporalConv(nn.Module):
    r"""Applies a factored 3D convolution over an input signal composed of several input 
    planes with distinct spatial and time axes, by performing a 2D convolution over the 
    spatial axes to an intermediate subspace, followed by a 1D convolution over the time 
    axis to produce the final output.
    Args:
        in_channels (int): Number of channels in the input tensor
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to the sides of the input during their respective convolutions. Default: 0
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(SpatioTemporalConv, self).__init__()

        # if ints are entered, convert them to iterables, 1 -> [1, 1, 1]
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)

        # decomposing the parameters into spatial and temporal components by
        # masking out the values with the defaults on the axis that
        # won't be convolved over. This is necessary to avoid unintentional
        # behavior such as padding being added twice
        spatial_kernel_size =  [1, kernel_size[1], kernel_size[2]]
        spatial_stride =  [1, stride[1], stride[2]]
        spatial_padding =  [0, padding[1], padding[2]]

        temporal_kernel_size = [kernel_size[0], 1, 1]
        temporal_stride =  [stride[0], 1, 1]
        temporal_padding =  [padding[0], 0, 0]

        # compute the number of intermediary channels (M) using formula 
        # from the paper section 3.5
        intermed_channels = int(math.floor((kernel_size[0] * kernel_size[1] * kernel_size[2] * in_channels * out_channels)/ \
                            (kernel_size[1]* kernel_size[2] * in_channels + kernel_size[0] * out_channels)))

        # the spatial conv is effectively a 2D conv due to the 
        # spatial_kernel_size, followed by batch_norm and ReLU
        self.spatial_conv = nn.Conv3d(in_channels, intermed_channels, spatial_kernel_size,
                                    stride=spatial_stride, padding=spatial_padding, bias=bias)
        self.bn = nn.BatchNorm3d(intermed_channels)
        self.relu = nn.ReLU()

        # the temporal conv is effectively a 1D conv, but has batch norm 
        # and ReLU added inside the model constructor, not here. This is an 
        # intentional design choice, to allow this module to externally act 
        # identical to a standard Conv3D, so it can be reused easily in any 
        # other codebase
        self.temporal_conv = nn.Conv3d(intermed_channels, out_channels, temporal_kernel_size, 
                                    stride=temporal_stride, padding=temporal_padding, bias=bias)

    def forward(self, x):
        x = self.relu(self.bn(self.spatial_conv(x)))
        x = self.temporal_conv(x)
        return x
    


if (__name__ == '__main__'):
    options = {"model": {'numclasses': 32}}
    data = torch.zeros((4, 3, 18, 112, 112))
    m = Dense3D('')
    # for k, v in m.state_dict().items():
    #     print(k)
    print(m)
    print(m(data).size())
