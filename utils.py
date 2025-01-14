'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
    - count_parameters: number of parameters of input model
'''
import os
import sys
import time
import torch
import shutil

import torch.nn as nn
import torch.nn.init as init
import torchvision
import torchvision.transforms as transforms

from models import *
from datasets import tinyimagenet, imagenette
from models.smallnet import SmallNet


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model(net_name, num_classes):
    if net_name == 'ResNet18':
        return ResNet18(num_classes)
    elif net_name == 'ResNet34':
        return ResNet34(num_classes)
    elif net_name == 'MobileNetV2':
        return MobileNetV2(num_classes=num_classes)
    elif net_name == 'SENet18':
        return SENet18(num_classes=num_classes)
    elif net_name == 'PreActResNet18':
        return PreActResNet18(num_classes=num_classes)
    elif net_name == 'DenseNet121':
        return DenseNet121(num_classes=num_classes)
    elif net_name == 'LeNet':
        return LeNet(num_classes=num_classes)
    elif net_name == 'GoogLeNet':
        return GoogLeNet(num_classes=num_classes)
    elif net_name == 'ShuffleNet':
        return ShuffleNetV2(num_classes=num_classes)
    elif net_name == 'VGG':
        return VGG(vgg_name='VGG11', num_classes=num_classes)
    elif net_name == 'NIN':
        return NIN(num_classes=num_classes)
    elif net_name == 'AlexNet':
        return AlexNet(num_classes=num_classes)
    elif net_name == 'SmallNet':
        return SmallNet(num_classes=num_classes)


def get_dataset(dataset, batch_size):
    print('==> Preparing data..')

    loaded_dataset, trainset, testset = None, None, None
    if dataset == 'CIFAR10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(10),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    elif dataset == 'CIFAR100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(10),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    elif dataset == 'MNIST':
        transform_train = transforms.Compose([
            transforms.Resize((32, 32)),
            torchvision.transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor()
        ])

        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            torchvision.transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor()
        ])

        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)

    elif dataset == 'FashionMNIST':
        transform_train = transforms.Compose([
            transforms.Resize((32, 32)),
            torchvision.transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            torchvision.transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_test)

    elif dataset == 'TinyImageNet':
        transform_train = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        transform_test = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        trainset = tinyimagenet.TinyImageNet(root='./data', split='train', download=True, transform=transform_train)
        testset = tinyimagenet.TinyImageNet(root='./data', split='val', download=True, transform=transform_test)

    elif dataset == 'ImageNette':
        transform_train = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        transform_test = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        trainset = imagenette.ImageNette(root='./data', split='train', download=True, transform=transform_train)
        testset = imagenette.ImageNette(root='./data', split='val', download=True, transform=transform_test)

    elif dataset == 'ImageWoof':
        transform_train = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        transform_test = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        trainset = imagenette.ImageWoof(root='./data', split='train', download=True, transform=transform_train)
        testset = imagenette.ImageWoof(root='./data', split='val', download=True, transform=transform_test)

    # If we have train, test sets already
    if loaded_dataset is None:
        num_of_classes = len(trainset.classes)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)
    # If we have to split the dataset
    else:
        pass

    num_batches = len(trainset) / batch_size
    print('Dataset:', trainset.__class__.__name__, 'Batch size:', batch_size)
    return trainloader, testloader, num_batches, num_of_classes


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


# if os.name != 'nt':
#     _, term_width = shutil.get_terminal_size()
#     _, term_width = os.popen('stty size', 'r').read().split()
#     term_width = int(term_width)

term_width = 80
TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
