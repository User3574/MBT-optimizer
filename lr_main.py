# -*- coding: utf-8 -*-
"""
Accompanying implementation for paper:
BACKTRACKING GRADIENT DESCENT METHOD FOR GENERAL C1 FUNCTIONS, WITH APPLICATIONS TO DEEP LEARNING
https://arxiv.org/pdf/1808.05160.pdf

Forked and inspired by https://github.com/kuangliu/pytorch-cifar
Train CIFAR10 with PyTorch
Learning rate finder using Backtracking line search with different batch sizes
and different starting learning rates
"""
# from log import backup, login, logout; backup(); login()
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import os
import json, pickle, click

from models import *
from utils import count_parameters, get_dataset, get_model
from lr_backtrack import LRFinder


@click.command()
@click.option('--lr_justified', default=True, help='Is lr justified')
@click.option('--alpha', default=1e-4, help='Hyperparameter alpha')
@click.option('--beta', default=0.5, help='Hyperparameter beta')
@click.option('--num_iter', default=20, help='Number of iterations')
@click.option('--resume', default=False, help='Resume from checkpoint')
@click.option('--net_name', default='ResNet18',
              help="Choose specific network architecture: "
                   "ResNet18, ResNet34, MobileNetV2, SENet18, PreActResNet18, DenseNet121, LeNet, "
                   "GoogLeNet, ShuffleNet, VGG, NIN, AlexNet")
@click.option('--dataset', default='FashionMNIST', required=True, help='Dataset to evaluate',
              type=click.Choice(['CIFAR10', 'CIFAR100', 'MNIST', 'FashionMNIST']))
def run_experiments(lr_justified, alpha, beta, num_iter, resume, net_name, dataset):
    all_batch_sizes = [12, 25, 50, 100, 200, 400, 800]
    all_lr_starts = [100, 10, 1, 0.1, 0.01, 0.001, 1e-4, 1e-5, 1e-6]
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Data
    _, _, _, num_classes = get_dataset(dataset, 128)
    
    save_paths = ['weights/', 'history', 'history/lr']
    for save_path in save_paths:
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
    # Model
    net = get_model(net_name, num_classes)
    
    print('Model:', net_name)
    print('Number of parameters:', count_parameters(net), 'Numbers of Layers:', len(list(net.parameters())))
    net = net.to(device)
    
    save_dir = save_path + net_name + '/'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    weights_best = save_dir + net_name + '_d' + str(dataset) + '_best.t7'

    # cuda device
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    
    if resume:
        # Load checkpoint.
        print('Resuming from checkpoint %s..' % weights_best)
        assert os.path.isfile(weights_best), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(weights_best)
        net.load_state_dict(checkpoint['net'])
    
    criterion = nn.CrossEntropyLoss()
    
    lr_full = {}
    # loop for batch size
    for batch_size in all_batch_sizes:
        lr_full[batch_size] = {}
        trainloader, testloader, num_batches, num_classes = get_dataset(dataset, batch_size)
        # loop for starting learning rate
        for lr_start in all_lr_starts:
            optimizer_BT = optim.SGD(net.parameters(), lr=lr_start)
            print('Start learning rate:', optimizer_BT.param_groups[0]['lr'])

            lr_finder_BT = LRFinder(net, optimizer_BT, criterion, device="cuda")
            print("Using backtrack with", optimizer_BT.__class__.__name__, ", alpha =", alpha, ', beta =', beta)

            lr_finder_BT.backtrack(trainloader, alpha=alpha, beta=beta, num_iter=num_iter, lr_justified=lr_justified)
            lr_full[batch_size][lr_start] = lr_finder_BT.lr_BT

            json.dump(lr_full, open(f"history/lr/lr_full{str(net_name)}_d{str(dataset)}.json", 'w'), indent=4)
            pickle.dump(lr_full, open(f"history/lr/lr_full{str(net_name)}_d{str(dataset)}.pickle", 'wb'))
    
    # print result
    print('Learning rate finding result using Backtracking line search with different batch sizes:')
    print(lr_full)


if __name__ == '__main__':
    run_experiments()
