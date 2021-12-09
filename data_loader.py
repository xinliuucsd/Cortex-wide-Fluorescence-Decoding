import torch
from torch.utils import data
import numpy as np
import h5py
from scipy.io import loadmat


class Dataset(data.Dataset):
    def __init__(self, input_data_file,
                 img_data_file,
                 f_interval,
                 e_interval,
                 shuffle=False,
                 ):
        if shuffle:
            np.random.shuffle(input_data_file)
        self.input_data_file = input_data_file  # ECoG
        self.img_data_file = img_data_file  # image data
        self.f_interval = f_interval
        self.e_interval = e_interval

    def __getitem__(self, index):
        if index < self.f_interval:
            X = None
            y = None
        elif index >= self.img_data_file.shape[0] - self.e_interval:
            X = None
            y = None
        if index >= self.f_interval and index < self.img_data_file.shape[0] - self.e_interval:
            X = self.input_data_file[index - self.f_interval: index + self.e_interval, :]
            y = self.img_data_file[index, :]
        return X, y

    def __len__(self):
        return self.input_data_file.shape[0]


def my_collate_fn(batch):
    """
    Custom collate function to ignore the None data samples
    """
    data_sample = [item[0] for item in batch if item[0] is not None]
    target = [item[1] for item in batch if item[1] is not None]
    data_sample = torch.cuda.FloatTensor(data_sample)
    target = torch.cuda.LongTensor(target)

    return [data_sample, target]


def loading_matfile(file_name, key, key2=''):
    """
    @ param: string: file_name
    @ param: string: key
    return: key data in numpy.array form
    """
    if file_name is None:
        print("Invalid file name!")
    # res = loadmat(file_name)[key]
    f = h5py.File(file_name, 'r')
    res = f.get(key + "/" + key2)
    return res
