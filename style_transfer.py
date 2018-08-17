import copy
import time
import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
from torch.utils.data import *
import matplotlib.pyplot as plt

from util import helpers as helper
from util import loaders as load
from models import networks as n

plt.switch_backend('agg')


############################################################################
# Train
############################################################################

class StyleTransfer:
    """
    Example usage if not using command line:

    from style_transfer import *
    params = {'dataset': 'VOCdevkit/VOC2007/JPEGImages/',
              'in_channels': 3,
              'batch_size': 4,
              'gen_filters': 128,
              'gen_layers':2,
              'layers_c' :2,
              'layers_s' :[3, 8, 15, 22],
              'style_weight': 10000000000,
              'content_weight': 10000,
              'res_blocks':5,
              'lr_gen': 1e-3,
              'test_perc': .1,
              'data_perc': 1,
              'beta1': .5,
              'beta2': .999,
              'train_epoch': 60,
              'res':256,
              'save_every': 5,
              'ids': [16, 26],
              'workers':4,
              'style_image': 'style/franc.jpg',
              'save_root': 'franc_style'}
    st = StyleTransfer(params)
    st.train()
    """

    def __init__(self, params):
        self.params = params
        self.model_dict = {}
        self.opt_dict = {}
        self.current_epoch = 0
        self.current_iter = 0

        # Setup data loaders
        self.transform = load.NormDenorm([.485, .456, .406],
                                         [.229, .224, .225])

        self.train_data = load.GenericDataset(params['dataset'],
                                              self.transform,
                                              output_res=params["res"],
                                              test_perc=params["test_perc"],
                                              data_perc=params["data_perc"])

        self.data_len = self.train_data.__len__()
        self.test_data = copy.deepcopy(self.train_data)
        self.test_data.train = False

        self.train_loader = torch.utils.data.DataLoader(self.train_data,
                                                        batch_size=params["batch_size"],
                                                        num_workers=params["workers"],
                                                        shuffle=True,
                                                        drop_last=True)

        self.test_loader = torch.utils.data.DataLoader(self.test_data,
                                                       batch_size=params["batch_size"],
                                                       num_workers=params["workers"],
                                                       shuffle=True,
                                                       drop_last=True)
        print('Data Loaders Initialized')
        # Setup models
        self.model_dict["G"] = n.Generator(layers=params["gen_layers"],
                                           filts=params["gen_filters"],
                                           channels=params["in_channels"],
                                           res_layers=params["res_blocks"])

        self.tensor_transform = n.TensorTransform(res=params["res"],
                                                  mean=[.485, .456, .406],
                                                  std=[.229, .224, .225])
        self.tensor_transform.cuda()

        for i in self.model_dict.keys():
            self.model_dict[i].apply(helper.weights_init_normal)
            self.model_dict[i].cuda()
            self.model_dict[i].train()

        print('Networks Initialized')
        # Setup loss
        self.style = load.open_style(params["style_image"],
                                     self.transform,
                                     batch_size=params["batch_size"],
                                     size=params["res"]).cuda()

        self.vgg = n.make_vgg()
        self.vgg.cuda()

        self.cs_loss = n.ContStyleLoss(self.vgg, params['layers_s'],
                                       self.style, params['content_weight'],
                                       params['style_weight'],
                                       params['layers_c'])

        # Setup optimizers
        self.opt_dict["G"] = optim.Adam(self.model_dict["G"].parameters(),
                                        lr=params['lr_gen'],
                                        betas=(params['beta1'], params['beta2']))

        print('Losses Initialized')

        # Setup history storage
        self.losses = ['S_Loss', 'C_Loss']
        self.loss_batch_dict = {}
        self.loss_batch_dict_test = {}
        self.loss_epoch_dict = {}
        self.loss_epoch_dict_test = {}
        self.train_hist_dict = {}
        self.train_hist_dict_test = {}

        for loss in self.losses:
            self.train_hist_dict[loss] = []
            self.loss_epoch_dict[loss] = []
            self.loss_batch_dict[loss] = []
            self.train_hist_dict_test[loss] = []
            self.loss_epoch_dict_test[loss] = []
            self.loss_batch_dict_test[loss] = []

    def load_state(self, filepath):
        # Load previously saved sate from disk, including models, optimizers and history
        state = torch.load(filepath)
        self.current_iter = state['iter'] + 1
        self.current_epoch = state['epoch'] + 1

        for i in self.model_dict.keys():
            self.model_dict[i].load_state_dict(state['models'][i])
        for i in self.opt_dict.keys():
            self.opt_dict[i].load_state_dict(state['optimizers'][i])

        self.train_hist_dict = state['train_hist']
        self.train_hist_dict_test = state['train_hist_test']

    def save_state(self, filepath):
        # Save current state of all models, optimizers and history to disk
        out_model_dict = {}
        out_opt_dict = {}
        for i in self.model_dict.keys():
            out_model_dict[i] = self.model_dict[i].state_dict()
        for i in self.opt_dict.keys():
            out_opt_dict[i] = self.opt_dict[i].state_dict()

        model_state = {'iter': self.current_iter,
                       'epoch': self.current_epoch,
                       'models': out_model_dict,
                       'optimizers': out_opt_dict,
                       'train_hist': self.train_hist_dict,
                       'train_hist_test': self.train_hist_dict_test
                       }

        torch.save(model_state, filepath)
        return f'Saving State at Iter:{self.current_iter}'

    def display_history(self):
        # Draw history of losses, called at end of training
        fig = plt.figure()
        for key in self.losses:
            x = range(len(self.train_hist_dict[key]))
            x_test = range(len(self.train_hist_dict_test[key]))
            if len(x) > 0:
                plt.plot(x, self.train_hist_dict[key], label=key)
                plt.plot(x_test, self.train_hist_dict_test[key], label=key + '_test')

        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend(loc=2)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'output/{self.params["save_root"]}_loss.jpg')
        plt.show()
        plt.close(fig)

    def train_gen(self, real_default, real_vgg):
        # train function for generator
        self.opt_dict["G"].zero_grad()

        fake = self.model_dict["G"](real_default)
        fake = self.tensor_transform(fake)
        ct_losses, st_losses = self.cs_loss(fake, real_vgg)

        self.loss_batch_dict['S_Loss'] = sum(st_losses) / self.params['batch_size']
        self.loss_batch_dict['C_Loss'] = sum(ct_losses) / self.params['batch_size']

        # Step
        total_loss = self.loss_batch_dict['S_Loss'] + self.loss_batch_dict['C_Loss']
        total_loss.backward()
        self.opt_dict["G"].step()
        return st_losses, ct_losses

    def test_gen(self, real_default, real_vgg):
        # test function for generator
        fake = self.model_dict["G"](real_default)
        fake = self.tensor_transform(fake)
        ct_losses, st_losses = self.cs_loss(fake, real_vgg)

        self.loss_batch_dict_test['S_Loss'] = sum(st_losses) / self.params['batch_size']
        self.loss_batch_dict_test['C_Loss'] = sum(ct_losses) / self.params['batch_size']
        return st_losses, ct_losses

    def test_loop(self):
        # Test on validation set
        self.model_dict["G"].eval()

        for loss in self.losses:
            self.loss_epoch_dict_test[loss] = []
        # test loop #
        for real_vgg, real_default in tqdm(self.test_loader):
            real_vgg = Variable(real_vgg.cuda())
            real_default = Variable(real_default.cuda())

            # TEST GENERATOR
            style_losses, content_losses = self.test_gen(real_default, real_vgg)

            # append all losses in loss dict #
            [self.loss_epoch_dict_test[loss].append(self.loss_batch_dict_test[loss].item()) for loss in self.losses]
        [self.train_hist_dict_test[loss].append(helper.mft(self.loss_epoch_dict_test[loss])) for loss in self.losses]

    def train(self):
        # Train following learning rate schedule
        params = self.params
        while self.current_epoch < params["train_epoch"]:
            # clear last epochs losses
            for loss in self.losses:
                self.loss_epoch_dict[loss] = []

            self.model_dict["G"].train()
            epoch_start_time = time.time()
            num_iter = 0

            print(f"Sched Sched Iter:{self.current_iter}, Sched Epoch:{self.current_epoch}")
            [print(f"Learning Rate({opt}): {self.opt_dict[opt].param_groups[0]['lr']}") for opt in self.opt_dict.keys()]
            for real_vgg, real_default in tqdm(self.train_loader):
                real_vgg = Variable(real_vgg.cuda())
                real_default = Variable(real_default.cuda())

                # TRAIN GENERATOR
                style_losses, content_losses = self.train_gen(real_default, real_vgg)

                # append all losses in loss dict
                [self.loss_epoch_dict[loss].append(self.loss_batch_dict[loss].item()) for loss in self.losses]
                self.current_iter += 1
                num_iter += 1

            # generate test images and save to disk
            helper.show_test(params,
                             self.transform,
                             self.tensor_transform,
                             self.test_loader,
                             self.style,
                             self.model_dict['G'],
                             save=f'output/{params["save_root"]}_val_{self.current_epoch}.jpg')

            # run validation set loop to get losses
            self.test_loop()
            if self.current_epoch % params["save_every"] == 0:
                save_str = self.save_state(f'output/{params["save_root"]}_{self.current_epoch}.json')
                tqdm.write(save_str)

            self.current_epoch += 1
            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time
            print(f'Epoch Training Training Time: {per_epoch_ptime}')
            [print(f'Train {loss}: {helper.mft(self.loss_epoch_dict[loss])}') for loss in self.losses]
            [print(f'Val {loss}: {helper.mft(self.loss_epoch_dict_test[loss])}') for loss in self.losses]
            print('\n')
            [self.train_hist_dict[loss].append(helper.mft(self.loss_epoch_dict[loss])) for loss in self.losses]

        self.display_history()
        print('Hit End of Learning Schedule!')
