'''Load data
'''

from __future__ import print_function
import os
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def load_data(data_type, config):

    transform_train = transforms.Compose([
        transforms.Resize(config['image_size']),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.RandomApply([AddGaussianNoise(std=0.1)], p=0.2),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_test = transforms.Compose([
        transforms.Resize(config['image_size']),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    train_path = os.path.join(config['train_path'], data_type)
    test_path = os.path.join(config['test_path'], 'Test_'+data_type)

    train_data = datasets.ImageFolder(train_path, transform_train)
    test_data = datasets.ImageFolder(test_path, transform_test)

    train_loader = DataLoader(train_data, batch_size=config['batch_size'],
            shuffle=True)
    test_loader = DataLoader(test_data, batch_size=config['batch_size'],
            shuffle=False)

    return train_data, test_data, train_loader, test_loader

