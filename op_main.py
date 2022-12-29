"""
Accompanying implementation for paper:
BACKTRACKING GRADIENT DESCENT METHOD FOR GENERAL C1 FUNCTIONS, WITH APPLICATIONS TO DEEP LEARNING
https://arxiv.org/pdf/1808.05160.pdf

Train CIFAR10 with PyTorch using Resnet18 with optimizers with different starting learning rates
Forked and inspired by https://github.com/kuangliu/pytorch-cifar
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import numpy as np
import os

from models import *
from utils import count_parameters, get_dataset, get_model
from optimizers.inna import INNA
from tqdm import tqdm

from lr_backtrack import LRFinder, change_lr
import json, pickle, click


@click.command()
@click.option('--start_epoch', default=0, help='Starting epoch')
@click.option('--batch_size', default=200, help='Batch size')
@click.option('--lr_start', default=100, help='Start learning rate')
@click.option('--momentum', default=0.9, help='Momentum size')
@click.option('--resume', default=False, help='Resume from checkpoint')
@click.option('--net_name', default='ResNet18',
              help="Choose specific network architecture: "
                   "ResNet18, ResNet34, MobileNetV2, SENet18, PreActResNet18, DenseNet121, LeNet, "
                   "GoogLeNet, ShuffleNet, VGG, NIN")
@click.option('--dataset', default='CIFAR10', required=True, help='Dataset to evaluate',
              type=click.Choice(['CIFAR10', 'CIFAR100', 'MNIST', 'FashionMNIST']))
def run_experiments(start_epoch, batch_size, lr_start, momentum, resume, net_name, dataset):
    global best_loss, loss_avg, history, patient_test, patient_train, patient, optimizer, apply, alpha, best_acc
    all_lr_starts = [100, 10, 1, 0.1, 0.01, 0.001, 1e-4, 1e-5, 1e-6]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # number of epochs waiting for improvement of best_acc or best_loss
    patient = 0

    # Data
    trainloader, testloader, num_batches, num_classes = get_dataset(dataset, batch_size)

    # Model
    net = get_model(net_name, num_classes)
    print('Model:', net_name)
    print('Number of parameters:', count_parameters(net), 'Numbers of Layers:', len(list(net.parameters())))
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()

    # Weights Directory
    save_paths = ['weights/', 'history', 'history/optimizers']
    for save_path in save_paths:
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

    save_dir = save_path + net_name + '/'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    weights_init = save_dir + net_name + '_CF' + str(num_classes) + '_init.t7'  # initial weights path
    weights_best = save_dir + net_name + '_CF' + str(num_classes) + '_best.t7'  # best weights path
    history_path = save_dir + net_name + '_CF' + str(num_classes) + '_history.json'  # history path

    # CUDA device
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    # if resuming
    if resume:
        # Load checkpoint.
        print('Resuming from checkpoint %s..' % weights_best)
        assert os.path.isfile(weights_best), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(weights_best)
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    # list of optimizers
    optimizers = {
        'SGD': optim.SGD(net.parameters(), lr=lr_start),
        'MMT': optim.SGD(net.parameters(), lr=lr_start, momentum=momentum),
        'NAG': optim.SGD(net.parameters(), lr=lr_start, momentum=momentum, nesterov=True),
        'Adagrad': optim.Adagrad(net.parameters(), lr=lr_start),
        'Adadelta': optim.Adadelta(net.parameters(), lr=lr_start),
        'RMSprop': optim.RMSprop(net.parameters(), lr=lr_start),
        'Adam': optim.Adam(net.parameters(), lr=lr_start),
        'Adamax': optim.Adamax(net.parameters(), lr=lr_start),
        'INNA': INNA(net.parameters(), lr=lr_start)
    }

    all_history = {}
    # loop for different starting learning rates and optimizers
    for lr_start in all_lr_starts:
        all_history[lr_start] = {}
        for op_name in optimizers:
            patient_train = 0
            patient_test = 0
            patient = 0
            best_acc = 0  # best test accuracy
            best_loss = loss_avg = 1e10  # best (smallest) training loss
            optimizer = optimizers[op_name]
            change_lr(optimizer, lr_start)
            print('Optimizer:', op_name, ', start learning rate:', optimizer.param_groups[0]['lr'])

            if not resume:
                if os.path.isfile(weights_init):
                    print('Loading initialized weights from %s' % weights_init)
                    net.load_state_dict(torch.load(weights_init))
                else:
                    print('Saving initialized weights to %s' % weights_init)
                    torch.save(net.state_dict(), weights_init)

            history = {'lr': [], 'acc_train': [], 'acc_valid': [], 'loss_train': [], 'loss_valid': []}

            # Training
            def train(epoch):
                global best_loss, loss_avg, history, patient_train, patient_train, patient, optimizer
                train_loss = correct = total = 0
                patient = min([patient_test, patient_train])
                print('\nEpoch: %d' % epoch)

                net.train()
                pbar = tqdm(enumerate(trainloader))
                for batch_idx, (inputs, targets) in pbar:
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()
                    outputs = net(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                    acc = 100. * correct / total
                    loss_avg = train_loss / (batch_idx + 1)

                    pbar.set_description(f"Batch: %d/%d | Loss: %.3f | Acc: %.3f%% (%d/%d)| LR: %.7f" %
                                         (batch_idx, len(trainloader), loss_avg, acc, correct, total,
                                          optimizer.param_groups[0]['lr']))

                history['acc_train'].append(acc)
                history['loss_train'].append(loss_avg)
                history['lr'].append(optimizer.param_groups[0]['lr'])

                if loss_avg > best_loss or np.isnan(loss_avg):
                    patient_train += 1
                    print('Total training loss does not decrease in last %d epoch(s)' % (patient_train))
                else:
                    patient_train = 0
                    best_loss = loss_avg

            # Testing
            def test(epoch):
                global history, patient_train, patient_test, best_acc
                net.eval()
                test_loss = 0
                correct = 0
                total = 0
                with torch.no_grad():
                    pbar = tqdm(enumerate(testloader))
                    for batch_idx, (inputs, targets) in pbar:
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = net(inputs)
                        loss = criterion(outputs, targets)

                        test_loss += loss.item()
                        _, predicted = outputs.max(1)
                        total += targets.size(0)
                        correct += predicted.eq(targets).sum().item()

                        acc = 100. * correct / total
                        loss_avg = test_loss / (batch_idx + 1)

                        pbar.set_description(f"Batch: %d/%d | Loss: %.3f | Acc: %.3f%% (%d/%d)" %
                                             (batch_idx, len(testloader), loss_avg, 100. * correct / total, correct,
                                              total))

                history['acc_valid'].append(acc)
                history['loss_valid'].append(loss_avg)

                # Save checkpoint.
                acc = 100. * correct / total

                if acc > best_acc:
                    patient_test = 0
                    print('Best valid accuracy!')
                    best_acc = acc
                else:
                    patient_test += 1
                    print('Total valid accuracy does not increase in last %d epoch(s)' % (patient_test))

            # Main loop for training and testing with early stopping
            for epoch in range(start_epoch, 200):
                if patient < 50:
                    train(epoch)
                    test(epoch)

                    all_history[lr_start][op_name] = history
                    json.dump(history, open(history_path, 'w'), indent=4)
                    json.dump(all_history, open("history/optimizers/all_optimizers_cifar%d.json" % (num_classes), 'w'),
                              indent=4)
                    pickle.dump(all_history,
                                open("history/optimizers/all_optimizers_cifar%d.pickle" % (num_classes), 'wb'))


if __name__ == '__main__':
    run_experiments()
