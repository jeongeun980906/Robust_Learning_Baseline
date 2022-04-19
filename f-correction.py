import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
import math
import torchvision.models as models
import random
import os

from dataloader.cifar import CIFAR10, CIFAR100
from dataloader.mnist import MNIST
from dataloader.dirty_mnist import DirtyMNIST
from dataloader.dirty_cifar import dirtyCIFAR10
from dataloader.clothing1m import clothing1M
from model.cnn import CNN,CNN2
import torchvision.models as models

import numpy as np
from matplotlib import pyplot as plt
import sys,json
from f_correction.utils import *
import torch.nn as nn

def save_checkpoint(state, filename='./f_correction/ckpt/checkpoint.pth.tar'):
    torch.save(state, filename)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='This is the official implementation for the ICML 2019 Unsupervised label noise modeling and loss correction paper. This work is under MIT licence. Please refer to the RunScripts.sh and README.md files for example usages. Consider citing our work if this code is usefull for your project')
    parser.add_argument('--gpu', type = int, default = 0)
    parser.add_argument('--root-dir', type=str, default='.', help='path to CIFAR dir where cifar-10-batches-py/ and cifar-100-python/ are located. If the datasets are not downloaded, they will automatically be and extracted to this path, default: .')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='input batch size for training, default: 128')
    parser.add_argument('--test-batch-size', type=int, default=100,
                        help='input batch size for testing, default: 100')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs to train, default: 10')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='initial learning rate, default: 0.1')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist','cifar10','dirtymnist','dirtycifar10','clothing1m'], help='dataset to train on, default: CIFAR10')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum, default: 0.9')
    
    parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.2)
    parser.add_argument('--noise_type', type = str, help='[pairflip, symmetric]', default='symmetric')
    parser.add_argument('--experiment-name', type=str, default='runs',
                        help='name of the experiment for the output files storage, default: runs')
    parser.add_argument('--alpha', type=float, default=32, help='alpha parameter for the mixup distribution, default: 32')
    parser.add_argument('--M', nargs='+', type=int, default=[100, 150],
                        help="Milestones for the LR sheduler, default 100 250")
    parser.add_argument('--Mixup', type=str, default='None', choices=['None', 'Static', 'Dynamic'],
                        help="Type of bootstrapping. Available: 'None' (deactivated)(default), \
                                'Static' (as in the paper), 'Dynamic' (BMM to mix the smaples, will use decreasing softmax), default: None")
    parser.add_argument('--BootBeta', type=str, default='Hard', choices=['None', 'Hard', 'Soft'],
                        help="Type of Bootstrapping guided with the BMM. Available: \
                        'None' (deactivated)(default), 'Hard' (Hard bootstrapping), 'Soft' (Soft bootstrapping), default: Hard")
    parser.add_argument('--reg-term', type=float, default=0., 
                        help="Parameter of the regularization term, default: 0.")


    args = parser.parse_args()
    
    print(args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu) 
    torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode
    torch.manual_seed(0)  # CPU seed
    torch.cuda.manual_seed_all(0)  # GPU seed

    random.seed(0)  # python seed for image transformation

    device = torch.device('cuda:{}'.format(args.gpu))
    if args.dataset=='mnist':
        input_channel=1
        num_classes=10
        args.top_bn = False
        args.epoch_decay_start = 5
        args.n_epoch = 20
        trainset = MNIST(root='./data/',
                                    download=True,  
                                    train=True, 
                                    transform=transforms.ToTensor(),
                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate
                            )
        trainset_track = MNIST(root='./data/',
                                    download=True,  
                                    train=True, 
                                    transform=transforms.ToTensor(),
                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate
                            )
        
        testset = MNIST(root='./data/',
                                download=True,  
                                train=False, 
                                transform=transforms.ToTensor(),
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                            )
    if args.dataset=='dirtymnist':
        input_channel=1
        num_classes=10
        args.top_bn = False
        args.epoch_decay_start = 5
        args.n_epoch = 20
        trainset = DirtyMNIST(root='./data/',
                                    train=True, 
                                    # transform=transforms.ToTensor(),
                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate
                            )
        trainset_track = DirtyMNIST(root='./data/',
                                    train=True, 
                                    # transform=transforms.ToTensor(),
                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate
                            )
        
        testset = DirtyMNIST(root='./data/',
                                train=False, 
                                # transform=transforms.ToTensor(),
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                            )

    if args.dataset=='cifar10':
        input_channel=3
        num_classes=10
        args.top_bn = False
        args.epoch_decay_start = 80
        args.n_epoch = 200

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

        trainset = CIFAR10(root='./data/',
                                    download=True,  
                                    train=True, 
                                    transform=transform_train,
                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate
                            )
        trainset_track=CIFAR10(root='./data/',
                                    download=True,  
                                    train=True, 
                                    transform=transform_train,
                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate
                            )
        testset = CIFAR10(root='./data/',
                                    download=True,  
                                    train=False, 
                                    transform=transform_test,
                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate
                            )

    if args.dataset=='dirtycifar10':
        input_channel=3
        num_classes=10
        args.top_bn = False
        args.epoch_decay_start = 80
        args.n_epoch = 200

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

        trainset = dirtyCIFAR10(root='./data/',
                                    train=True,  
                                    transform=transform_train,
                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate
                            )
        trainset_track=dirtyCIFAR10(root='./data/',
                                    train=True,  
                                    transform=transform_train,
                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate
                            )
        testset = dirtyCIFAR10(root='./data/',
                                    train=False, 
                                    transform=transform_test,
                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate
                            )
    
    if args.dataset == 'clothing1m':
        input_channel=3
        num_classes=14
        args.top_bn = False
        args.epoch_decay_start = 40
        args.n_epoch = 80
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

        trainset = clothing1M(
            root        = '/home/sungjoon.choi/seungyoun/Clothing1M',
            transform   = transform_train,
            mode        = 'train',
            num_samples = 1000*args.batch_size
        )

        trainset_track = clothing1M(
            root        = '/home/sungjoon.choi/seungyoun/Clothing1M',
            transform   = transform_train,
            mode        = 'train',
            num_samples = 1000*args.batch_size
        )

        testset = clothing1M(
            root        = '/home/sungjoon.choi/seungyoun/Clothing1M',
            transform   = transform_val,
            mode        = 'test'
        )


    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    train_loader_track = torch.utils.data.DataLoader(trainset_track, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    if args.dataset=='cifar10' or args.dataset=='dirtycifar10' or args.dataset == 'cifar100':
        gamma = 0.5
        model = CNN(n_outputs=10).to(device)
        milestones = args.M
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-6)

    elif args.dataset=='mnist' or args.dataset=='dirtymnist':
        gamma = 0.5
        model=CNN2(n_outputs=10).to(device)
        milestones = [10,15]
        optimizer = optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-6,eps=1e-8)
    else:
        milestones = [40]
        gamma = 0.1
        model =models.resnet50(pretrained=True)
        model.fc = nn.Linear(2048,num_classes)
        model = model.to(device)
        optimizer = optim.SGD(model.parameters(),lr=2e-3,weight_decay=1e-3,momentum=0.9)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    # path where experiments are saved
    exp_path = os.path.join('./f_correction/ckpt', '{}_{}'.format(args.noise_type, args.noise_rate))

    if not os.path.isdir(exp_path):
         os.makedirs(exp_path)

    bmm_model=bmm_model_maxLoss=bmm_model_minLoss=cont=k = 0

    bootstrap_ep_std = milestones[0] + 5 + 1 # the +1 is because the conditions are defined as ">" or "<" not ">="
    guidedMixup_ep = 106

    if args.Mixup == 'Dynamic':
        bootstrap_ep_mixup = guidedMixup_ep + 5
    else:
        bootstrap_ep_mixup = milestones[0] + 5 + 1

    countTemp = 1

    temp_length = 200 - bootstrap_ep_mixup

    test_acc_list=list()
    train_acc_list=list()

    for epoch in range(1, args.epochs + 1):
        # train
        scheduler.step()

        ### Standard CE training (without mixup) ###
        if args.Mixup == "None":
            print('\t##### Doing standard training with cross-entropy loss #####')
            loss_per_epoch, acc_train_per_epoch_i = train_CrossEntropy(args, model, device, train_loader, optimizer, epoch)

        ### Mixup ###
        if args.Mixup == "Static":
            alpha = args.alpha
            if epoch < bootstrap_ep_mixup:
                print('\t##### Doing NORMAL mixup for {0} epochs #####'.format(bootstrap_ep_mixup - 1))
                loss_per_epoch, acc_train_per_epoch_i = train_mixUp(args, model, device, train_loader, optimizer, epoch, 32)

            else:
                if args.BootBeta == "Hard":
                    print("\t##### Doing HARD BETA bootstrapping and NORMAL mixup from the epoch {0} #####".format(bootstrap_ep_mixup))
                    loss_per_epoch, acc_train_per_epoch_i = train_mixUp_HardBootBeta(args, model, device, train_loader, optimizer, epoch,\
                                                                                    alpha, bmm_model, bmm_model_maxLoss, bmm_model_minLoss, args.reg_term, num_classes)
                elif args.BootBeta == "Soft":
                    print("\t##### Doing SOFT BETA bootstrapping and NORMAL mixup from the epoch {0} #####".format(bootstrap_ep_mixup))
                    loss_per_epoch, acc_train_per_epoch_i = train_mixUp_SoftBootBeta(args, model, device, train_loader, optimizer, epoch, \
                                                                                    alpha, bmm_model, bmm_model_maxLoss, bmm_model_minLoss, args.reg_term, num_classes)

        ## Dynamic Mixup ##
        if args.Mixup == "Dynamic":
            alpha = args.alpha
            if epoch < guidedMixup_ep:
                print('\t##### Doing NORMAL mixup for {0} epochs #####'.format(guidedMixup_ep - 1))
                loss_per_epoch, acc_train_per_epoch_i = train_mixUp(args, model, device, train_loader, optimizer, epoch, 32)

            elif epoch < bootstrap_ep_mixup:
                print('\t##### Doing Dynamic mixup from epoch {0} #####'.format(guidedMixup_ep))
                loss_per_epoch, acc_train_per_epoch_i = train_mixUp_Beta(args, model, device, train_loader, optimizer, epoch, alpha, bmm_model,\
                                                                        bmm_model_maxLoss, bmm_model_minLoss)
            else:
                print("\t##### Going from SOFT BETA bootstrapping to HARD BETA with linear temperature and Dynamic mixup from the epoch {0} #####".format(bootstrap_ep_mixup))
                loss_per_epoch, acc_train_per_epoch_i, countTemp, k = train_mixUp_SoftHardBetaDouble(args, model, device, train_loader, optimizer, \
                                                                                                                epoch, bmm_model, bmm_model_maxLoss, bmm_model_minLoss, \
                                                                                                                countTemp, k, temp_length, args.reg_term, num_classes)
        ### Training tracking loss
        epoch_losses_train, epoch_probs_train, argmaxXentropy_train, bmm_model, bmm_model_maxLoss, bmm_model_minLoss = \
            track_training_loss(args, model, device, train_loader_track, epoch, bmm_model, bmm_model_maxLoss, bmm_model_minLoss)

        # test
        loss_per_epoch, acc_val_per_epoch_i = test_cleaning(args, model, device, test_loader)
        test_acc=acc_val_per_epoch_i[-1].tolist()
        print('best_epoch_%d_valLoss_%.5f_valAcc_%.5f' % (
                epoch, loss_per_epoch[-1], acc_val_per_epoch_i[-1]))
        cont+=1
        test_acc_list.append(test_acc)
        train_acc_list.append(acc_train_per_epoch_i[-1])
        if epoch == args.epochs:
            snapLast = 'last_epoch_%d_valLoss_%.5f_valAcc_%.5f' % (
                epoch, loss_per_epoch[-1], acc_val_per_epoch_i[-1])
            torch.save(model.state_dict(), os.path.join(exp_path, snapLast + '.pth'))
            torch.save(optimizer.state_dict(), os.path.join(exp_path, 'opt_' + snapLast + '.pth'))
    if args.dataset != 'clothing1m':
        filename = '{}_{}_{}.json'.format(args.dataset,args.noise_type,args.noise_rate)
    else:
        filename = '{}.json'.format(args.dataset)
    log_data = {'train_acc' : train_acc_list,
                    'test_acc' : test_acc_list}
    with open("./f_correction/log/"+filename, "w") as json_file:
            json.dump(log_data,json_file)

if __name__ == '__main__':
    main()