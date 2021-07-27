import torch
import torch.nn as nn
import torch.nn.functional as F

class SeparableConv2d(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels, 
                 kernel_size = 1, 
                 stride = 1,
                 padding = 0, 
                 dilation = 1,
                 bias = False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias = bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias = bias)
    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, 
                 in_filters, 
                 out_filters,
                 kernel_size,
                 stride,
                 reps, 
                 bn = True,
                 act_fn = nn.ReLU,
                 start_with_act=True, 
                 grow_first=True):
        super().__init__()
        self.act_fn = act_fn
        if out_filters != in_filters or stride != 1 or stride != (1,1):
            skip = [nn.Conv2d(in_filters, out_filters, 1, stride = stride, bias = False)]
            if bn:
                skip.append(nn.BatchNorm2d(out_filters))
            self.skip = nn.Sequential(*skip)
        else:
            self.skip = None
        
        if isinstance(kernel_size, int) == False and len(kernel_size) > 1:
            p1 = (kernel_size[1] // 2, kernel_size[1] // 2 - (kernel_size[1] - 1) % 2)
            p2 = (kernel_size[0] // 2, kernel_size[0] // 2 - (kernel_size[0] - 1) % 2)
        else:
            p1 = (kernel_size // 2, kernel_size // 2 - (kernel_size - 1) % 2)
            p2 = p1
        padding_size = p1 + p2
        
        rep = []
        filters = in_filters
        if grow_first:
            rep.append(self.act_fn())
            rep.append(nn.ZeroPad2d(padding_size))
            rep.append(SeparableConv2d(in_filters, out_filters, kernel_size, stride = 1, padding = 0, bias = False))
            if bn:
                rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters
        for i in range(reps-1):
            rep.append(self.act_fn())
            rep.append(nn.ZeroPad2d(padding_size))
            rep.append(SeparableConv2d(filters, filters, kernel_size, stride = 1, padding = 0, bias = False))
            if bn:
                rep.append(nn.BatchNorm2d(filters))
        if not grow_first:
            rep.append(self.act_fn())
            rep.append(nn.ZeroPad2d(padding_size))
            rep.append(SeparableConv2d(in_filters, out_filters,kernel_size,stride=1,padding=0,bias=False))
            if bn:
                rep.append(nn.BatchNorm2d(out_filters))
        
        if not start_with_act:
            rep = rep[1:]
        else:
            rep[0] = self.act_fn()
        
        if stride != (1,1) or stride != 1:
            rep.append(nn.ZeroPad2d(padding_size))
            rep.append(nn.MaxPool2d(kernel_size, stride, 0))
        self.rep = nn.Sequential(*rep)
    def forward(self, inp):
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skip(inp)
        else:
            skip = inp
        x += skip
        return x
    
class Xception(nn.Module):
    """
    Paper is from https://arxiv.org/pdf/1610.02357.pdf
    code is from 
    """
    def __init__(self,
                 input_channel = 12,
                 act_fn = 'relu',
                 num_classes = 71,
                 bn = True,
                 list_kernel_size = [(3,3)] * 16,
                 list_strides = [(2,2), (2,2), (2,2), (2,2), (2,2)]
                ):
        """
        Args:
            input_channel: input channel dimension
            num_classes: number of classes
            kernel_size: kernel_sizes (some 2-d data can be (30,5000))
            list_strides: list of strides - (some 2-d data can be (30,5000))
                *** strides are for entry flow in conv1, block1, block2, block3, block12
        """
        super().__init__()
        self.input_channel = input_channel
        self.num_classes = num_classes
        self.list_kernel_size = list_kernel_size
        self.list_strides = list_strides
        self.bn = bn
        if act_fn == 'relu':
            self.act_fn = nn.ReLU
        elif act_fn == 'gelu':
            self.act_fn = nn.GELU
        ### ENTRY FLOW
        self.conv1 = nn.Conv2d(input_channel, 32, list_kernel_size[0], list_strides[0], 0, bias = False)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = self.act_fn()
        
        self.conv2 = nn.Conv2d(32, 64, list_kernel_size[1], 1, bias = False)
        self.bn2 = nn.BatchNorm2d(64)
        self.act2 = self.act_fn()
        
        self.block1 = Block(in_filters = 64, out_filters = 128, kernel_size = list_kernel_size[2], stride = list_strides[1], reps = 2, bn = bn, act_fn = self.act_fn, start_with_act = False, grow_first = True)
        self.block2 = Block(in_filters = 128, out_filters = 256, kernel_size = list_kernel_size[3], stride = list_strides[2], reps = 2, bn = bn, act_fn = self.act_fn, start_with_act = True, grow_first = True)
        self.block3 = Block(in_filters = 256, out_filters = 728, kernel_size = list_kernel_size[4], stride = list_strides[3], reps = 2, bn = bn, act_fn = self.act_fn, start_with_act = True, grow_first = True)
        
        
        ### MIDDLE FLOW
        self.block4 = Block(in_filters = 728, out_filters = 728, kernel_size = list_kernel_size[5], stride = 1, reps = 3, bn = bn, act_fn = self.act_fn, start_with_act = True, grow_first = True)
        self.block5 = Block(in_filters = 728, out_filters = 728, kernel_size = list_kernel_size[6], stride = 1, reps = 3, bn = bn, act_fn = self.act_fn, start_with_act = True, grow_first = True)
        self.block6 = Block(in_filters = 728, out_filters = 728, kernel_size = list_kernel_size[7], stride = 1, reps = 3, bn = bn, act_fn = self.act_fn, start_with_act = True, grow_first = True)
        self.block7 = Block(in_filters = 728, out_filters = 728, kernel_size = list_kernel_size[8], stride = 1, reps = 3, bn = bn, act_fn = self.act_fn, start_with_act = True, grow_first = True)
        
        self.block8 = Block(in_filters = 728, out_filters = 728, kernel_size = list_kernel_size[9], stride = 1, reps = 3, bn = bn, act_fn = self.act_fn, start_with_act = True, grow_first = True)
        self.block9 = Block(in_filters = 728, out_filters = 728, kernel_size = list_kernel_size[10], stride = 1, reps = 3, bn = bn, act_fn = self.act_fn, start_with_act = True, grow_first = True)
        self.block10 = Block(in_filters = 728, out_filters = 728, kernel_size = list_kernel_size[11], stride = 1, reps = 3, bn = bn, act_fn = self.act_fn, start_with_act = True, grow_first = True)
        self.block11 = Block(in_filters = 728, out_filters = 728, kernel_size = list_kernel_size[12], stride = 1, reps = 3, bn = bn, act_fn = self.act_fn, start_with_act = True, grow_first = True)

                
        ### EXIT FLOW
        self.block12 = Block(in_filters = 728, out_filters = 1024, kernel_size = list_kernel_size[13], stride = list_strides[4], reps = 2, bn = bn, act_fn = self.act_fn, start_with_act = True, grow_first = False)
                
        self.conv3 = SeparableConv2d(1024, 1536, list_kernel_size[14], 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.act3 = self.act_fn()
        
        self.conv4 = SeparableConv2d(1536, 2048, list_kernel_size[15], 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)
        
        self.fc = nn.Linear(2048, num_classes)
        self.last_linear = self.fc
    def features(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.act1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        return x
    def logits(self, features):
        # this is relu4
        x = self.act_fn()(features)
        
        x = F.adaptive_avg_pool2d(x, (1,1))
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x
    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

def xception(
    input_channel = 12, 
    act_fn = 'relu',
    num_classes = 71, 
    bn = True,
    list_kernel_size = [(3,3)] * 16, 
    list_strides = [(2,2), (2,2), (2,2), (2,2), (2,2)],
    pretrained = False, 
    model_dir = None
):
    model = Xception(input_channel, act_fn, num_classes, bn, list_kernel_size, list_strides)
    if pretrained and model_dir is not None:
        model.load_state_dict(torch.load(model_dir))
    return model