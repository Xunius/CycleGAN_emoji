'''Train cyclegan model

Author: guangzhi XU (xugzhi1987@gmail.com)
Update time: 2022-07-22 12:39:04.
'''

from __future__ import print_function
import os
import shutil
import json
import numpy as np
import torch
from torchvision.utils import save_image, make_grid

try:
    import torch.utils.tensorboard as tensorboard
    HAS_TENSORBOARD = True
except:
    HAS_TENSORBOARD = False

import loader
import model as model_module
from config import config


def generate_eval(model, test_X, test_Y, n_eval_samples, epoch, output_dir, device, nrow=2):

    save_dir = os.path.join(output_dir, 'samples')
    os.makedirs(save_dir, exist_ok=True)

    img_XY = []
    img_YX = []

    print('Generating eval samples...')
    for ii in range(n_eval_samples):
        idx = np.random.randint(len(test_X))
        idy = np.random.randint(len(test_Y))

        img_X, _ = test_X[idx]
        img_Y, _ = test_Y[idy]
        fake_X = gen_samples(model.gen_Y2X, img_Y, device)
        fake_Y = gen_samples(model.gen_X2Y, img_X, device)

        img_XY.extend([img_X, fake_Y])
        img_YX.extend([img_Y, fake_X])

    imgs = img_XY + img_YX
    imgs = [xx.squeeze().to('cpu') * 0.5 + 0.5 for xx in imgs]
    imgs = make_grid(imgs, nrow=nrow * n_eval_samples)

    img_path = os.path.join(save_dir, 'epoch_%s.png' %(str(epoch)))
    save_image(imgs, img_path)

    return

def gen_samples(gen_model, img, device):

    img = img.unsqueeze(0).to(device)
    gen_model.eval()
    with torch.no_grad():
        fake = gen_model(img)
    return fake

def train(epoch, total_iters, train_loader_X, train_loader_Y, test_X,
        test_Y, model, config, device, tb_writer=None):

    model.to_device(device)
    print('Epoch:', epoch, 'Device:', device)

    iterator = zip(iter(train_loader_X), iter(train_loader_Y))
    d_loss_total = 0
    g_loss_total = 0
    n_iter = 0

    for ii, (sample_X, sample_Y) in enumerate(iterator):

        total_iters += 1
        n_iter += 1
        img_X, _ = sample_X
        img_Y, _ = sample_Y
        img_X = img_X.to(device)
        img_Y = img_Y.to(device)

        d_loss, g_loss = model.train(total_iters, img_X, img_Y, config)

        #------------------Print progress------------------
        d_loss_total += d_loss.item()
        g_loss_total += g_loss.item()

        dis_lr = model.dis_scheduler.get_last_lr()[0]
        gen_lr = model.gen_scheduler.get_last_lr()[0]

        if tb_writer is not None:
            tb_writer.add_scalar('iters/g_loss', g_loss.item(), total_iters)
            tb_writer.add_scalar('iters/d_loss', d_loss.item(), total_iters)
            tb_writer.add_scalar('iters/g_lr', gen_lr, total_iters)
            tb_writer.add_scalar('iters/d_lr', dis_lr, total_iters)

        if ii % config['n_iters_eval'] == 0:
            print('epoch: %d, tot_iter: %d, dloss: %.3f, gloss: %.3f, d_lr: %.4f, g_lr: %.4f'\
                    %(epoch, total_iters, d_loss.item(), g_loss.item(), dis_lr, gen_lr))

            generate_eval(model, test_X, test_Y,
                    config['n_eval_samples'], epoch,
                    config['output_dir'], device)

    return total_iters, d_loss_total/n_iter, g_loss_total/n_iter


#-------------Main---------------------------------
if __name__=='__main__':


    model = model_module.CycleGAN(config)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to_device(DEVICE)

    if HAS_TENSORBOARD:
        log_dir = os.path.join(config['output_dir'], 'logs')
        os.makedirs(log_dir, exist_ok=True)
        tb_writer = tensorboard.SummaryWriter(log_dir=log_dir)

    config_path = os.path.join(config['output_dir'], 'exp_config.txt')
    with open(config_path, 'w') as fout:
        json.dump(config, fout, indent=4)

    shutil.copy2(model_module.__file__, os.path.join(config['output_dir'], 'model.py'))
    print('Copied model def file into', config['output_dir'])

    if config['resume']:
        start_epoch = config['resume_from_epoch']
        total_iters = model.load_checkpoint(config['output_dir'], epoch=start_epoch) + 1
        start_epoch += 1
    else:
        start_epoch = 0
        total_iters = 0

    train_dataX, test_dataX, train_loaderX, test_loaderX = loader.load_data('Apple', config)
    train_dataY, test_dataY, train_loaderY, test_loaderY = loader.load_data('Windows', config)


    for e in range(start_epoch, start_epoch+config['n_epochs']):


        total_iters, d_loss, g_loss = train(e, total_iters,
                train_loaderX, train_loaderY,
                test_dataX, test_dataY, model, config, DEVICE, tb_writer)

        if HAS_TENSORBOARD:
            tb_writer.add_scalar('epoch/g_loss', g_loss, e)
            tb_writer.add_scalar('epoch/d_loss', d_loss, e)

        if e % config['n_epoch_save'] == 0:
            model.save_checkpoint(config['output_dir'], e, total_iters)

    model.save_checkpoint(config['output_dir'], e, total_iters)
