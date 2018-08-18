from torch.utils.data import *
import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision.models as models


############################################################################
# Re-usable blocks
############################################################################


class TensorTransform(nn.Module):
    # Used to convert between default color space and VGG colorspace
    def __init__(self, res=256, mean=[.485, .456, .406], std=[.229, .224, .225]):
        super(TensorTransform, self).__init__()

        self.mean = torch.zeros([3, res, res]).cuda()
        self.mean[0, :, :] = mean[0]
        self.mean[1, :, :] = mean[1]
        self.mean[2, :, :] = mean[2]

        self.std = torch.zeros([3, res, res]).cuda()
        self.std[0, :, :] = std[0]
        self.std[1, :, :] = std[1]
        self.std[2, :, :] = std[2]

    def forward(self, x):
        norm_ready = (x * .5) + .5
        result = (norm_ready - self.mean) / self.std
        return result


class ConvTrans(nn.Module):
    # One Block to be used as conv and transpose throughout the model
    def __init__(self, ic=4, oc=4, kernel_size=4, block_type='up', padding=1):
        super(ConvTrans, self).__init__()
        self.block_type = block_type

        operations = []
        if self.block_type == 'up':
            operations += [nn.Upsample(mode='nearest', scale_factor=2),
                           nn.ReflectionPad2d(padding),
                           nn.Conv2d(in_channels=ic, out_channels=oc, kernel_size=kernel_size, stride=1, bias=True)]
        elif self.block_type == 'down':
            operations += [nn.ReflectionPad2d(padding),
                           nn.Conv2d(in_channels=ic, out_channels=oc, kernel_size=kernel_size, stride=2, bias=True)]

        operations += [nn.InstanceNorm2d(oc, affine=True)]
        operations += [nn.ReLU(inplace=True)]

        self.operations = nn.Sequential(*operations)

    def forward(self, x):
        x = self.operations(x)
        return x


class ResBlock(nn.Module):
    # Res Block for the center of the model
    def __init__(self, ic=4, oc=4, kernel_size=3, dropout=.5, use_dropout=False):
        super(ResBlock, self).__init__()

        stride = 1

        operations = []
        operations += [nn.ReflectionPad2d(1)]
        operations += [
            nn.Conv2d(in_channels=ic, out_channels=oc, padding=0, kernel_size=kernel_size, stride=stride, bias=True)]
        operations += [nn.InstanceNorm2d(oc, affine=True)]
        operations += [nn.ReLU(inplace=True)]

        if use_dropout:
            operations += [nn.Dropout(dropout)]

        operations += [nn.ReflectionPad2d(1)]
        operations += [
            nn.Conv2d(in_channels=ic, out_channels=oc, padding=0, kernel_size=kernel_size, stride=stride, bias=True)]
        operations += [nn.InstanceNorm2d(oc, affine=True)]

        self.block = nn.Sequential(*operations)

    def forward(self, x):
        return (x * .2) + self.block(x)


############################################################################
# Generator
############################################################################


class Generator(nn.Module):
    # Generator grown from smallest layer
    def __init__(self, layers=3, filts=1024, kernel_size=3, channels=3, res_layers=3):
        super(Generator, self).__init__()

        # residual core
        operations = [ResBlock(ic=filts, oc=filts, use_dropout=False, kernel_size=3) for i in range(res_layers)]

        # conv and trans building out from core
        for a in range(layers):
            down = [ConvTrans(ic=int(filts // 2), oc=filts, kernel_size=kernel_size, block_type='down')]
            up = [ConvTrans(ic=filts, oc=int(filts // 2), kernel_size=kernel_size, block_type='up')]
            operations = down + operations + up
            filts = int(filts // 2)

        # our input and our output
        inp = [nn.ReflectionPad2d(4),
               nn.Conv2d(in_channels=channels, out_channels=filts, padding=0, kernel_size=9, stride=1),
               nn.InstanceNorm2d(filts, affine=True),
               nn.ReLU(inplace=True)]
        out = [nn.ReflectionPad2d(4),
               nn.Conv2d(in_channels=filts, out_channels=channels, padding=0, kernel_size=9, stride=1)]

        operations = inp + operations + out
        self.model = nn.Sequential(*operations)

    def forward(self, x):
        return F.tanh(self.model(x))


############################################################################
# VGG Net
############################################################################


def make_vgg():
    vgg = models.vgg16(pretrained=True)
    children = list(vgg.children())
    children.pop()
    vgg = children[0]
    vgg.eval()
    for param in vgg.parameters():
        param.requires_grad = False
    return vgg


############################################################################
# Hook and Losses
############################################################################


def gram_mse_loss(mse_input, target):
    # Calculate MSE loss between two gram matrices
    return F.mse_loss(gram_matrix(mse_input), gram_matrix(target))


def gram_matrix(gram_input):
    # Calcule Gram Matrix
    b, c, h, w = gram_input.size()
    x = gram_input.view(b, c, -1)
    return torch.bmm(x, x.transpose(1, 2)) / (c * h * w)


class SetHook:
    # Register hook inside of network to retrieve features
    feats = None

    def __init__(self, block):
        self.hook_reg = block.register_forward_hook(self.hook)

    def hook(self, module, hook_input, output):
        self.feats = output

    def close(self):
        self.hook_reg.remove()


class ContStyleLoss(nn.Module):
    # Store Features for style, and calculate style and content loss
    def __init__(self, vgg, style_im, ct_wgt, style_wgt, style_layer_ids, content_layer_id):
        super().__init__()
        self.index = content_layer_id
        self.m, self.ct_wgt, self.style_wgt = vgg, ct_wgt, style_wgt
        self.sfs = [SetHook(vgg[i]) for i in style_layer_ids]
        vgg(style_im)
        self.style_feat = [o.feats.data.clone() for o in self.sfs]

    def forward(self, input_img, target_img):
        self.m(target_img.data)
        targ_feat = self.sfs[self.index].feats.data.clone()
        self.m(input_img)
        inp_feat = [o.feats for o in self.sfs]

        result_ct = [F.mse_loss(inp_feat[self.index], targ_feat) * self.ct_wgt]
        result_st = [gram_mse_loss(inp, targ) * self.style_wgt for inp, targ in zip(inp_feat, self.style_feat)]

        return result_ct, result_st

    def close(self):
        [o.remove() for o in self.sfs]

