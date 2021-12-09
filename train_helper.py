import argparse
import numpy as np
from scipy.io import savemat

from torch.utils import data
from torch.autograd import Variable

import torch
from data_loader import loading_matfile, my_collate_fn

def gen_hyperparam(n_epochs,
                    batch_size,
                    lr, b1, b2,
                    n_cpu,
                    loss_name, model_name, pred_name,
                    f_interval, e_interval):
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=n_epochs, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=batch_size, help="size of the batches")
    parser.add_argument("--lr", type=float, default=lr, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=b1, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=b2, help="adam: decay of second order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=n_cpu, help="number of cpu threads to use during batch generation")

    parser.add_argument("--loss_name", type= str, default=loss_name, help="folder for loss values")
    parser.add_argument("--model_name", type= str, default=model_name, help="folder for learned models")
    parser.add_argument("--pred_name", type= str, default=pred_name, help="folder for prediction results")
    parser.add_argument("--f_interval", type= int, default= f_interval, help="front interval for data preparation")
    parser.add_argument("--e_interval", type= int, default= e_interval, help="end interval for data preparation")

    return parser.parse_args()


def eval_model(model, dataloader, loss_func):
    model = model.eval()
    loss_val = 0
    y_pred = []
    dataloader.shuffle = False
    for i, (x, imgs) in enumerate(dataloader):
        with torch.no_grad():
            real_imgs = Variable(imgs.type(torch.cuda.FloatTensor))
            input_x = Variable(x.type(torch.cuda.FloatTensor))
            gen_imgs = model(input_x)
            rnn_loss = loss_func(gen_imgs, real_imgs)
            loss_val += rnn_loss.item()
            y_pred.append(gen_imgs)
    y_pred = torch.cat(y_pred)
    loss_val = loss_val / len(dataloader)

    return y_pred, loss_val


def valid(model,
          x_valid, y_valid,
          criterion,
          f_interval, e_interval,
          batch_size = 64,
          ):
    """
    valid the trained model
    @ return: prediction, loss
    """
    model = model.eval()
    interval = e_interval + f_interval
    # convert to tensor
    x_valid = Variable(torch.from_numpy(x_valid).type(torch.cuda.FloatTensor))
    y_valid = Variable(torch.from_numpy(y_valid).type(torch.cuda.FloatTensor))
    # prepare validation dataset
    x_input = torch.zeros(x_valid.shape[0] - interval, interval, x_valid.shape[1])
    for i in range(0, x_valid.shape[0] - interval):
        x_input[i, :, :] = x_valid[i: i + interval, :]
        # x_input[i, :, :] = x_input[i, torch.randperm(90), :]  # Shuffle the data. Should be commented out for normal.

    with torch.no_grad():
        y_pred = torch.zeros(x_input.shape[0], y_valid.shape[1]).cuda()
        count = 0
        i = 0
        batch_size = batch_size
        while i < x_input.shape[0]:
            count += 1
            y_pred[i:i+ batch_size, :] = model(x_input[i:i+ batch_size].cuda())
            i +=  batch_size
        i -=  batch_size
        while i < x_input.shape[0]:
            y_pred[i, :] = model(x_input[i].unsqueeze(0).cuda())
            i += 1

        loss = []
        for k in range(y_pred.shape[1]):
            loss.append(criterion(y_pred[:, k], y_valid[f_interval : x_valid.shape[0] - e_interval, k]).item())

    return y_pred, loss


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.1)
    elif classname.find("BatchNorm1d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class data_prepare:
    # Configure data loader
    def __init__(self, img_data_file, ECoG_data_file, Dataset, img_feature='score', ecog_feature='ECoG_bands',
                 pc_id=1, total_fold = 5, f_interval=45, e_interval=45, **kwargs
                 ):
        # load the data
        img_data = loading_matfile(img_data_file, img_feature)
        ECoG_data = loading_matfile(ECoG_data_file, ecog_feature)
        if img_data.shape[0] < img_data.shape[1]:
            img_data = np.transpose(img_data)
        self.img_data = img_data[:, pc_id]
        if ECoG_data.shape[0] < ECoG_data.shape[-1]:
            ECoG_data = np.transpose(ECoG_data)
        # use specified frequency bands
        band_idx = kwargs.get('band_idx')
        ECoG_data = ECoG_data[:, band_idx, :]
        # use specified channels
        channel_idx = kwargs.get('channel_idx')
        ECoG_data = ECoG_data[:, :, channel_idx]
        # flatten the channel and frequency band axis
        self.ECoG_data = np.reshape(ECoG_data, (ECoG_data.shape[0], -1))
        len_ECoG_data = self.ECoG_data.shape[0]
        len_partition = len_ECoG_data // total_fold

        # prepare the train and validation dataset
        if kwargs.get('useBreak'):
            break1 = kwargs.get('break1')
            break2 = kwargs.get('break2')
            break1 = break1 * len_partition
            break2 = break2 * len_partition
            ind_all = np.arange(0,len_ECoG_data)
            valid_ind = np.arange(break1,break2)
            train_ind = np.setdiff1d(ind_all, valid_ind)

            self.ECoG_train_data = self.ECoG_data[train_ind, :]
            self.img_train_data = self.img_data[train_ind, :]
            self.ECoG_valid_data = self.ECoG_data[valid_ind]
            self.img_valid_data = self.img_data[valid_ind]
        else:
            self.img_train_data = self.img_data[0:len_partition, :]
            self.ECoG_train_data = self.ECoG_data[0:len_partition, :]
            self.img_valid_data = self.img_data[len_partition:-1, :]
            self.ECoG_valid_data = self.ECoG_data[len_partition:-1, :]

        self.Dataset = Dataset
        self.front_interval = f_interval
        self.end_interval = e_interval


    def get_len(self):

        img_train_data_shape =  self.img_train_data.shape
        ECoG_train_data_shape = self.ECoG_train_data.shape

        return "shape of img train data: {}, shape of ECoG train data: {}".format(img_train_data_shape, ECoG_train_data_shape )

    def get_train_dataloader(self, batch_size = 128, shuffle = False):
        train_dataset = self.Dataset(
                        input_data_file = self.ECoG_train_data,
                        img_data_file = self.img_train_data,
                        f_interval = self.front_interval,
                        e_interval = self.end_interval,
                        shuffle = shuffle
                        )
        dataloader = data.DataLoader(train_dataset,
                                     batch_size = batch_size,
                                     shuffle = True,
                                     collate_fn = my_collate_fn
                                      )
        return dataloader

    def get_valid_dataloader(self, batch_size = 128, shuffle = False):
        train_dataset = self.Dataset(
                        input_data_file = self.ECoG_valid_data,
                        img_data_file = self.img_valid_data,
                        f_interval = self.front_interval,
                        e_interval = self.end_interval,
                        shuffle=shuffle
                        )

        dataloader = data.DataLoader(train_dataset,
                                     batch_size = batch_size,
                                     shuffle = False,
                                     collate_fn = my_collate_fn
                                      )
        return dataloader

    def save_train_valid_data(self, save_path, pc_id):

        valid_dataset = {'input_data':self.ECoG_valid_data,
                          'img_data': self.img_valid_data}
        train_dataset = {'img_data': self.img_train_data,
                        'input_data':self.ECoG_train_data}
        file_name_valid = save_path + '/valid_truth.mat'
        file_name_train = save_path + '/train_truth.mat'
        savemat(file_name_valid,  {'truth_valid': valid_dataset['img_data'][:,:]})
        savemat(file_name_train,  {'truth_train': train_dataset['img_data'][:,:]})
        return train_dataset, valid_dataset

