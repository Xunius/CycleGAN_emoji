'''Use the cyclegan model to transfer emoji styles

Author: guangzhi XU (xugzhi1987@gmail.com)
Update time: 2022-07-22 12:39:04.
'''

from __future__ import print_function
import torch

from model import CycleGAN
from config import config
from loader import load_data
from train import generate_eval


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CycleGAN(config)
    model.to_device(device)

    # load the pre-trained weights
    model.load_checkpoint('./outputs/exp1', epoch=None)

    # load test data
    train_dataX, test_dataX, train_loaderX, test_loaderX = load_data('Apple', config)
    train_dataY, test_dataY, train_loaderY, test_loaderY = load_data('Windows', config)

    # do inference
    for _ in range(10):
        generate_eval(model, test_dataX, test_dataY, 5, 'latest_%d' %_, '.', device)



