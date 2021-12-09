import os
import time
import numpy as np
from scipy.io import savemat

import torch
from torch.autograd import Variable

from utils import save_loss
from data_loader import Dataset
from train_helper import valid, data_prepare, gen_hyperparam
from models import RNN_Final as model
#from torchinfo import summary

###################################################
# Settings for the decoding task
###################################################
total_fold = 10
fold = 1

input_feature = 'ECoG_band_norm_new_100'
output_feature = 'mean_dfof'  # specify "score_ica_norm" to decode ICA scores
pc_id = [0,1,2,3,4,5,6,7,8,9,10,11]  # index of IC scores to predict
band_idx = [0,1,2,3,4,5] # index of the frequency bands to use.
channel_idx = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] # index of the channels to use
matfile_name = 'Mouse2_Day1_run1_16Ch_ECoG_Ca_PC10_6_bands_new.mat' # recording session to use

if __name__ == "__main__":
    #################################################
    # Settings for training
    #################################################
    n_epochs = 20  # number of epochs to run
    batch_size = 128  # the batch size for each epoch
    # parameters for adam
    lr = 1e-4
    b1 = 0.9
    b2 = 0.999
    # data loader parameter
    n_cpu = 8
    # result folder
    saved_path = "saved_results/mouse2_day1_run1/mean_activity/fold" + str(fold) + '/'
    # cross validation number
    cross_val_folds = {}
    for i in range(0, total_fold):
        cross_val_folds[i + 1] = (i, i + 1)
    # input time steps
    f_interval = 45 # number of time steps before time t
    e_interval = 45 # number of time steps after time t. In total, these gives 90 time steps.
    # configure GPU
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(1)
    if torch.cuda.is_available():
        cuda = True
    else:
        cuda = False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    # output folders
    saved_loss = os.path.join(saved_path, "saved_loss")
    saved_model = os.path.join(saved_path, "saved_model")
    saved_data = os.path.join(saved_path, "saved_data")

    # configure hyper-parameters
    opt = gen_hyperparam(n_epochs=n_epochs, batch_size=batch_size, lr=lr, b1=b1, b2=b2, n_cpu=n_cpu,
                         loss_name=saved_loss, model_name=saved_model, pred_name=saved_data,
                         f_interval=f_interval, e_interval=e_interval)
    print(opt)

    # make folder if necessary
    os.makedirs(opt.loss_name, exist_ok=True)
    os.makedirs(opt.model_name, exist_ok=True)
    os.makedirs(opt.pred_name, exist_ok=True)

    # Specify the train - validation fold
    break1, break2 = cross_val_folds[fold]  # get the break1 and break2 from the cross_val_folds
    is_break = True  # use break to divide the cross-vaildation folders
    print("break 1: ", break1, "break 2: ", break2)

    # data preparation
    ECoG_data_file = '../Data/' + matfile_name
    img_data_file = '../Data/' + matfile_name
    data_set = data_prepare(img_data_file, ECoG_data_file, Dataset, output_feature, input_feature,
                            pc_id=pc_id, total_fold = total_fold,
                            f_interval=opt.f_interval, e_interval=opt.e_interval,
                            **{'break1': break1, 'break2': break2, 'useBreak': is_break,
                               'band_idx': band_idx,
                               'channel_idx': channel_idx
                               }
                            )
    dataloader_train = data_set.get_train_dataloader(opt.batch_size)
    # save the ground truth data for train and valid
    train_dataset, valid_dataset = data_set.save_train_valid_data(opt.pred_name, pc_id)
    print('train_dataset{}, valid_dataset{}'.format(train_dataset['input_data'].shape,
                                                    valid_dataset['input_data'].shape))
    print('len of train DataLoader {}'.format(len(dataloader_train)))
    print("batch size: ", opt.batch_size)

    ###################################################
    # Build the network, choose loss func and optimizer
    ###################################################
    ch_by_bands_num = train_dataset['input_data'].shape[1]
    rnn = model(interval=opt.f_interval + opt.e_interval, num_pc=len(pc_id), ch_by_bands_num = ch_by_bands_num)
    mse_loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)) # , weight_decay = 1)
    if cuda:
        rnn.cuda()
        mse_loss.cuda()
    # summary(rnn, input_size=(1, 90, 96))
    ###################################################
    # Start training and validation
    ###################################################

    for epoch in range(opt.n_epochs):
        # The training part
        train_loss = 0
        rnn.train()
        t0 = time.time()
        for i, (x, imgs) in enumerate(dataloader_train):
            real_imgs = Variable(imgs.type(Tensor))
            input_x = Variable(x.type(Tensor))
            # forward pass
            gen_imgs = rnn(input_x)
            rnn_loss = mse_loss(gen_imgs, real_imgs)
            train_loss += rnn_loss.item()
            # backward and optimize
            optimizer.zero_grad()
            rnn_loss.backward()
            optimizer.step()

            batches_done = epoch * len(dataloader_train) + i
        train_loss = train_loss / len(dataloader_train)
        # The validation part
        valid_pred, valid_loss = valid(rnn, valid_dataset['input_data'], valid_dataset['img_data'], mse_loss,
                                       f_interval=opt.f_interval, e_interval=opt.e_interval,
                                       batch_size=opt.batch_size)
        train_pred, train_loss = valid(rnn, train_dataset['input_data'], train_dataset['img_data'], mse_loss,
                                       f_interval=opt.f_interval, e_interval=opt.e_interval,
                                       batch_size=opt.batch_size)
        # save the prediction data for each epoch
        file_name_valid = opt.pred_name + '/valid_pred_epoch_{}.mat'.format(epoch)
        file_name_train = opt.pred_name + '/train_pred_epoch_{}.mat'.format(epoch)
        savemat(file_name_valid, {'valid': valid_pred.detach().cpu().numpy()})
        savemat(file_name_train, {'train': train_pred.detach().cpu().numpy()})
        # save the loss data for each epoch
        save_loss(train_loss, opt.loss_name + "/train_loss.txt", 'a')
        save_loss(valid_loss, opt.loss_name + "/valid_loss.txt", 'a')
        # Save the model for each epoch
        torch.save(rnn.state_dict(), os.path.join(opt.model_name, 'rnn_epoch_{}.pt'.format(epoch)))
        # Output some training progress
        t1 = time.time()
        print(
            "[Epoch %d/%d] [Batch %d/%d] [MSE train loss: %.4f][MSE valid loss: %.4f][time: %.2fs]"
            % (epoch, opt.n_epochs, i, len(dataloader_train), np.mean(train_loss), np.mean(valid_loss),
               t1 - t0)
        )