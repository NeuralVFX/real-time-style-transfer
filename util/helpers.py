import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import *
from torch.autograd import Variable


############################################################################
# Helper Utilities
############################################################################


def weights_init_normal(m):
    # Set initial state of weights
    classname = m.__class__.__name__
    if 'ConvTrans' == classname:
        pass
    elif 'Conv2d' in classname or 'Linear' in classname or 'ConvTrans' in classname:
        nn.init.normal_(m.weight.data, 0, .02)


def mft(tensor):
    # Return mean float tensor #
    return torch.mean(torch.FloatTensor(tensor))


############################################################################
# Display Images
############################################################################


def show_test(params, denorm, tensor_norm, dataloader, style, model, save=False):
    # Show and save
    ids_a = params['ids']
    image_grid_len = len(ids_a)
    fig, ax = plt.subplots(image_grid_len, 3, figsize=(13, 5*image_grid_len))
    count = 0
    model.eval()
    for idx, real in enumerate(dataloader):
        if idx in ids_a:
            real_vgg = Variable(real[0].cuda())
            real_def = Variable(real[1].cuda())
            test = tensor_norm(model(real_def))
            ax[count, 0].cla()
            ax[count, 0].imshow(denorm.denorm(real_vgg[0]))
            ax[count, 1].cla()
            ax[count, 1].imshow(denorm.denorm(test[0]))
            ax[count, 2].cla()
            ax[count, 2].imshow(denorm.denorm(style[0]))
            count += 1
    model.train()
    if save:
        plt.savefig(save)
    plt.show()
    plt.close(fig)
