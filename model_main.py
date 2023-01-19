"""
Accompanying implementation for paper:
BACKTRACKING GRADIENT DESCENT METHOD FOR GENERAL C1 FUNCTIONS, WITH APPLICATIONS TO DEEP LEARNING
https://arxiv.org/pdf/1808.05160.pdf

Forked and inspired by https://github.com/kuangliu/pytorch-cifar
Training different models on Cifar10 abd Cifar100 with MBT-MMT and MBT-NAG
"""

# from log import backup, login, logout; backup(); login()
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import os
import collections
import json
import click

from models import *
from tqdm import tqdm
from utils import count_parameters, get_dataset, get_model
from optimizers import inna
from lr_backtrack import LRFinder, change_lr


def get_backtracking_params(op_name):
    if op_name == 'SGD':
        alpha = 1e-4
        once_only = True
        apply = True
    elif op_name == 'MMT':
        alpha = 1e-4
        once_only = False
        apply = True
    elif op_name == 'NAG':
        alpha = 1e-4
        once_only = False
        apply = True
    else:
        return None, None, None
    return alpha, once_only, apply


@click.command()
@click.option('--start_epoch', default=0, help='Starting epoch')
@click.option('--batch_size', default=200, help='Batch size')
@click.option('--lr_start', default=100, help='Start learning rate')
@click.option('--use_backtracking', default=True, help='Using backtracking or not')
@click.option('--lr_justified', default=True, help='Is lr justified')
@click.option('--beta', default=0.5, help='Hyperparameter beta')
@click.option('--num_iter', default=20, help='Number of iterations')
@click.option('--momentum', default=0.9, help='Momentum size')
@click.option('--resume', default=False, help='Resume from checkpoint')
@click.option('--nets', default='LeNet', callback=lambda _, __, x: x.split(',') if x else [],
              help="Choose specific network architecture: "
                   "ResNet18, ResNet34, MobileNetV2, SENet18, PreActResNet18, DenseNet121, LeNet, "
                   "GoogLeNet, ShuffleNet, VGG, NIN")
@click.option('--dataset', default='FashionMNIST', required=True, help='Dataset to evaluate',
              type=click.Choice(['CIFAR10', 'CIFAR100', 'MNIST', 'FashionMNIST']))
def run_experiments(start_epoch, batch_size, lr_start, use_backtracking, lr_justified, beta, num_iter, momentum, resume, nets, dataset):
    global best_loss, loss_avg, history, patient_test, patient_train, patient, optimizer, apply, alpha, best_acc
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create save path
    save_paths = ['weights/', 'history', 'history/models/']
    for save_path in save_paths:
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
    # Data
    trainloader, testloader, num_batches, num_classes = get_dataset(dataset, batch_size)

    criterion = nn.CrossEntropyLoss()
    all_history = {}

    # Loop for model architetures
    for net_name in nets:
        net = get_model(net_name, num_classes)

        print('Dataset:', dataset)
        print('Model:', net_name)
        print('Number of parameters:', count_parameters(net), 'Numbers of Layers:', len(list(net.parameters())))
        net = net.to(device)

        save_dir = save_path + net_name + '/'
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        weights_init = save_dir + net_name + '_d' + str(dataset) + '_b' + str(batch_size) + '_init.t7'
        weights_best = save_dir + net_name + '_d' + str(dataset) + '_b' + str(batch_size) + '_best.t7'

        if device == 'cuda':
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True

        optimizers = {
            'SGD': optim.SGD(net.parameters(), lr=lr_start),
            'MMT': optim.SGD(net.parameters(), lr=lr_start, momentum=momentum),
            'NAG': optim.SGD(net.parameters(), lr=lr_start, momentum=momentum, nesterov=True),
        }
        # Loop for optimizers:
        for op_name in optimizers.keys():
            alpha, once_only, apply = get_backtracking_params(op_name)
            optimizer = optimizers[op_name]

            # Global variables
            history_path = save_dir + net_name + '_d' + str(dataset) + '_b' + str(batch_size) + '_history_MBT-' + op_name + '.json'
            patient_train = 0  # number of epochs waiting for improvement of training loss
            patient_test = 0  # number of epochs waiting for improvement of validation accuracy
            patient = 0  # basically min of patient_train and  patient_test
            best_acc = 0  # best test accuracy
            best_loss = loss_avg = 1e10  # best (smallest) training loss

            # Load checkpoint
            if resume:
                print('Resuming from checkpoint %s..' % weights_best)
                assert os.path.isfile(weights_best), 'Error: no checkpoint directory found!'
                checkpoint = torch.load(weights_best)
                net.load_state_dict(checkpoint['net'])
                best_acc = checkpoint['acc']
                start_epoch = checkpoint['epoch']

            print('Optimizer:', op_name, ', start learning rate:', optimizer.param_groups[0]['lr'])
            try:
                print('Momentum:', optimizer.param_groups[0]['momentum'], 'Nesterov:',
                      optimizer.param_groups[0]['nesterov'])
            except:
                print('No momentum')
            if not resume:
                if os.path.isfile(weights_init):
                    print('Loading initialized weights from %s' % weights_init)  # load initial weights
                    net.load_state_dict(torch.load(weights_init))
                else:
                    print('Saving initialized weights to %s' % weights_init)  # save initial weights
                    torch.save(net.state_dict(), weights_init)

            # Define gradient descent optimizer for backtracking process
            if use_backtracking:
                optimizer_BT = optim.SGD(net.parameters(), lr=lr_start)
                lr_finder_BT = LRFinder(net, optimizer_BT, criterion, device="cuda")

            history = {'lr': [], 'acc_train': [], 'acc_valid': [], 'loss_train': [], 'loss_valid': []}

            # Training
            def train(epoch):
                global best_loss, loss_avg, history, patient_test, patient_train, patient, optimizer, apply, alpha
                train_loss = correct = total = 0
                patient = min([patient_test, patient_train])

                if use_backtracking and (apply or patient > 5):
                    if patient > 5:
                        alpha = 0.5
                        optimizer = optim.SGD(net.parameters(), lr=optimizer.param_groups[0]['lr'])
                    print('\nEpoch: %d. Finding learning rate...' % epoch)
                    print("Using backtrack with", optimizer_BT.__class__.__name__, ", alpha =", alpha, ', beta =', beta)
                    change_lr(optimizer, lr_start)
                    lr_finder_BT.backtrack(trainloader, alpha=alpha, beta=beta, num_iter=num_iter,
                                           lr_justified=lr_justified)
                    optimizer.state = collections.defaultdict(dict)  # reset optimizer
                    change_lr(optimizer, lr_finder_BT.lr_BT)
                    apply = apply * (1 - once_only)
                else:
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

                if loss_avg > best_loss:
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
                    state = {
                        'net': net.state_dict(),
                        'acc': acc,
                        'epoch': epoch,
                    }

                    print('Best valid accuracy! Saving to %s...' % (weights_best))
                    torch.save(state, weights_best)
                    best_acc = acc
                else:
                    patient_test += 1
                    print('Total valid accuracy does not increase in last %d epoch(s)' % (patient_test))

            for epoch in range(start_epoch, 200):
                if patient < 50:
                    train(epoch)
                    test(epoch)
                    if op_name not in all_history:
                        all_history[op_name] = {}

                    all_history[op_name][lr_start] = history
                    json.dump(history, open(history_path, 'w'), indent=2)
                    json.dump(all_history, open(f"history/models/all_models{str(net_name)}_d{str(dataset)}_b{str(batch_size)}.pickle", 'w'),
                              indent=2)


if __name__ == '__main__':
    run_experiments()
