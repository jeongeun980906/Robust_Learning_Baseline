# -*- coding:utf-8 -*-
import os
import torch
import torchvision.transforms as transforms
from dataloader.cifar import CIFAR10, CIFAR100
from dataloader.mnist import MNIST
import argparse, sys
import datetime
from jocor.algorithm.jocor import JoCoR
import json
from dataloader.dirty_mnist import DirtyMNIST
from dataloader.dirty_cifar import dirtyCIFAR10
from dataloader.clothing1m import clothing1M

import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--result_dir', type=str, help='dir to save result txt files', default='results')
parser.add_argument('--noise_rate', type=float, help='corruption rate, should be less than 1', default=0.2)
parser.add_argument('--forget_rate', type=float, help='forget rate', default=None)
parser.add_argument('--noise_type', type=str, help='[pairflip, symmetric, asymmetric, asymmetric2, asymmetric3]', default='pairflip')
parser.add_argument('--num_gradual', type=int, default=10,
                    help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')
parser.add_argument('--exponent', type=float, default=1,
                    help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.')
parser.add_argument('--dataset', type=str, help='mnist, cifar10, or cifar100', default='mnist')
parser.add_argument('--n_epoch', type=int, default=200)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--num_iter_per_epoch', type=int, default=400)
parser.add_argument('--epoch_decay_start', type=int, default=80)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--co_lambda', type=float, default=0.47)
parser.add_argument('--adjust_lr', type=int, default=1)
parser.add_argument('--model_type', type=str, help='[mlp,cnn]', default='cnn')
parser.add_argument('--save_model', type=str, help='save model?', default="False")
parser.add_argument('--save_result', type=str, help='save result?', default="True")


args = parser.parse_args()

# Seed
torch.manual_seed(args.seed)
if args.gpu is not None:
    print('cuda')
    device = torch.device('cuda:{}'.format(args.gpu))
    torch.cuda.manual_seed(args.seed)

else:
    print('cpu')
    device = torch.device('cpu')
    torch.manual_seed(args.seed)

# Hyper Parameters
batch_size = 128
learning_rate = args.lr

# load dataset
if args.dataset == 'mnist':
    input_channel = 1
    num_classes = 10
    init_epoch = 1
    filter_outlier = True
    args.epoch_decay_start = 10
    args.model_type = "cnn2"
    args.n_epoch = 20
    train_dataset = MNIST(root='./data/',
                          download=True,
                          train=True,
                          transform=transforms.ToTensor(),
                          noise_type=args.noise_type,
                          noise_rate=args.noise_rate
                          )

    test_dataset = MNIST(root='./data/',
                         download=True,
                         train=False,
                         transform=transforms.ToTensor(),
                         noise_type=args.noise_type,
                         noise_rate=args.noise_rate
                         )

if args.dataset=='dirtymnist':
    input_channel = 1
    num_classes = 10
    init_epoch = 1
    filter_outlier = True
    args.epoch_decay_start = 10
    args.model_type = "cnn2"
    args.n_epoch = 20

    train_dataset = DirtyMNIST(root='./data/',
                                train=True, 
                                # transform=transforms.ToTensor(),
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                        )
    
    test_dataset = DirtyMNIST(root='./data/',
                            train=False, 
                            # transform=transforms.ToTensor(),
                            noise_type=args.noise_type,
                            noise_rate=args.noise_rate
                        )

if args.dataset=='cifar10':
    input_channel = 3
    num_classes = 10
    init_epoch = 5
    args.epoch_decay_start = 80
    # args.n_epoch = 200
    filter_outlier = False
    args.model_type = "cnn"

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.49137255, 0.48235294, 0.44666667), (0.24705882, 0.24352941, 0.26156863)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.49137255, 0.48235294, 0.44666667), (0.24705882, 0.24352941, 0.26156863)),
    ])

    train_dataset = CIFAR10(root='./data/',
                                download=True,  
                                train=True, 
                                transform=transform_train,
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                        )
    test_dataset = CIFAR10(root='./data/',
                                download=True,  
                                train=False, 
                                transform=transform_test,
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                        )

if args.dataset=='dirtycifar10':
    input_channel = 3
    num_classes = 10
    init_epoch = 5
    args.epoch_decay_start = 80
    args.n_epoch = 200
    filter_outlier = False
    args.model_type = "cnn"

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.49137255, 0.48235294, 0.44666667), (0.24705882, 0.24352941, 0.26156863)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.49137255, 0.48235294, 0.44666667), (0.24705882, 0.24352941, 0.26156863)),
    ])

    train_dataset = dirtyCIFAR10(root='./data/',
                                train=True, 
                                transform=transform_train,
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                        )
    test_dataset = dirtyCIFAR10(root='./data/',
                                train=False, 
                                transform=transform_test,
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate)
                           
if args.dataset == 'cifar100':
    input_channel = 3
    num_classes = 100
    init_epoch = 5
    args.epoch_decay_start = 100
    # args.n_epoch = 200
    filter_outlier = False
    args.model_type = "cnn"


    train_dataset = CIFAR100(root='./data/',
                             download=True,
                             train=True,
                             transform=transforms.ToTensor(),
                             noise_type=args.noise_type,
                             noise_rate=args.noise_rate
                             )

    test_dataset = CIFAR100(root='./data/',
                            download=True,
                            train=False,
                            transform=transforms.ToTensor(),
                            noise_type=args.noise_type,
                            noise_rate=args.noise_rate
                            )

if args.dataset == 'clothing1m':
    input_channel = 3
    num_classes = 14
    init_epoch = 2
    args.epoch_decay_start = 40
    args.n_epoch = 80
    filter_outlier = False
    args.model_type = "resnet"
    batch_size = 16 #32
    transform_train = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),                
                transforms.Normalize((0.6959, 0.6537, 0.6371),(0.3113, 0.3192, 0.3214)),                     
            ])    
    transform_val = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.6959, 0.6537, 0.6371),(0.3113, 0.3192, 0.3214)),
            ])  

    train_dataset = clothing1M(
        root        = '/home/sungjoon.choi/seungyoun/Clothing1M',
        transform   = transform_train,
        mode        = 'train',
        num_samples = 1000*batch_size
    )
    test_dataset = clothing1M(
        root        = '/home/sungjoon.choi/seungyoun/Clothing1M',
        transform   = transform_val,
        mode        = 'test'
    )

if args.forget_rate is None:
    forget_rate = args.noise_rate
else:
    forget_rate = args.forget_rate



def main():
    # Data Loader (Input Pipeline)
    train_acc1_list =list()
    train_acc2_list =list()
    test_acc1_list =list()
    test_acc2_list =list()
    os.makedirs("./log/"+args.dataset + "_" + args.noise_type + "(" + str(args.noise_rate) +")", exist_ok=True)

    print('loading dataset...')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               num_workers=args.num_workers,
                                               drop_last=True,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              num_workers=args.num_workers,
                                              drop_last=True,
                                              shuffle=False)
    # Define models
    print('building model...')

    model = JoCoR(args, train_dataset, device, input_channel, num_classes)

    epoch = 0
    train_acc1 = 0
    train_acc2 = 0

    # evaluate models with random weights
    test_acc1, test_acc2 = model.evaluate(test_loader)

    print(
        'Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f %% Model2 %.4f ' % (
            epoch + 1, args.n_epoch, len(test_dataset), test_acc1, test_acc2))


    acc_list = []
    # training
    for epoch in range(1, args.n_epoch):
        # train models
        train_acc1, train_acc2, pure_ratio_1_list, pure_ratio_2_list = model.train(train_loader, epoch)

        # evaluate models
        test_acc1, test_acc2 = model.evaluate(test_loader)

        train_acc1_list.append(train_acc1)
        train_acc2_list.append(train_acc2)
        test_acc1_list.append(test_acc1)
        test_acc2_list.append(test_acc2)

        # save results
        if pure_ratio_1_list is None or len(pure_ratio_1_list) == 0:
            print(
                'Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f %% Model2 %.4f' % (
                    epoch + 1, args.n_epoch, len(test_dataset), test_acc1, test_acc2))
        else:
            # save results
            mean_pure_ratio1 = sum(pure_ratio_1_list) / len(pure_ratio_1_list)
            mean_pure_ratio2 = sum(pure_ratio_2_list) / len(pure_ratio_2_list)
            print(
                'Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f %% Model2 %.4f %%, Pure Ratio 1 %.4f %%, Pure Ratio 2 %.4f %%' % (
                    epoch + 1, args.n_epoch, len(test_dataset), test_acc1, test_acc2, mean_pure_ratio1,
                    mean_pure_ratio2))


    #     if epoch >= 190:
    #         acc_list.extend([test_acc1, test_acc2])

    # avg_acc = sum(acc_list)/len(acc_list)
    # print(len(acc_list))
    # print("the average acc in last 10 epochs: {}".format(str(avg_acc)))
    
    # save model
    torch.save(model.model1.state_dict(), "./jocor/ckpt/"+args.dataset + "_" + args.noise_type + "(" + str(args.noise_rate) +").pth")
    torch.save(model.model2.state_dict(), "./jocor/ckpt/"+args.dataset + "_" + args.noise_type + "(" + str(args.noise_rate) +").pth")
    if args.dataset != 'clothing1m':
        filename = '{}_{}_{}.json'.format(args.dataset,args.noise_type,args.noise_rate)
    else:
        filename = '{}.json'.format(args.dataset)
    log_data = {'train_acc1' : train_acc1_list,
                    'train_acc2' : train_acc2_list,
                    'test_acc1' : test_acc1_list,
                    'test_acc2' : test_acc2_list}
    with open("./jocor/log/"+filename, "w") as json_file:
            json.dump(log_data,json_file)

if __name__ == '__main__':
    main()
