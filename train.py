from datetime import datetime
from functools import lru_cache

import numpy as np
import json, argparse

import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
from model.cnn import CNN,CNN2
from data.cifar import CIFAR10,CIFAR100
from data.mnist import MNIST
from total_variance.utils import get_transition,get_regularization,get_train_step,predict_test,predict_train,take_cycle,dirichlet_transition,kendall_tau

def train(args):
    # data
    # loaders
    batch_size = 128

    num_classes=10
    # iterations
    num_iter_total = 2000 if args.dataset=='mnist' else 4000
    num_iter_warmup = 400
    num_iter_test = int(num_iter_total / 10)

    # Seed
    torch.manual_seed(args.seed)
    if args.gpu is not None:
        print('cuda')
        device = torch.device('cuda:{}'.format(args.gpu))
    torch.cuda.manual_seed(args.seed)
    mnist = dict(
        device=device,
        lr=1e-3, lr_decay=0.1 ** (1 / num_iter_total),
    )
    cifar = dict(
        device=device, num_classes=num_classes,
        num_iter_warmup=num_iter_warmup, num_iter_total=num_iter_total,
        lr=0.1, momentum=0.9, weight_decay=1e-4,
    )

    # transition
    transition_type = 'none'
    categorical = dict(
        device=device, num_classes=num_classes,
        num_iter_warmup=num_iter_warmup, num_iter_total=num_iter_total,
        diagonal=np.log(0.5), off_diagonal=np.log(0.5 / (num_classes - 1)),
        lr=5e-3,
    )
    dirichlet = dict(
        device=device, num_classes=num_classes,
        diagonal=10. if args.dataset=='mnist'else 100., off_diagonal=0.,
        betas=(0.999, 0.01),
    )

    # regularization
    regularization_type = 'none'
    num_pairs = batch_size
    gamma = 0.1


    if args.dataset=='mnist':
        input_channel=1
        num_classes=10
        args.top_bn = False
        args.epoch_decay_start = 80
        args.n_epoch = 50
        trainset = MNIST(root='./data/',
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
        model = CNN2().to(device)
        optimizer = optim.Adam(model.parameters(), lr=mnist['lr'])
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=mnist['lr_decay'])
        
    if args.dataset=='cifar10':
        input_channel=3
        num_classes=10
        args.top_bn = False
        args.epoch_decay_start = 80
        args.n_epoch = 200
        trainset = CIFAR10(root='./data/',
                                    download=True,  
                                    train=True, 
                                    transform=transforms.ToTensor(),
                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate
                            )

        testset = CIFAR10(root='./data/',
                                    download=True,  
                                    train=False, 
                                    transform=transforms.ToTensor(),
                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate
                            )
        model = CNN().to(device)
        optimizer = optim.SGD(model.parameters(), lr=cifar['lr'], momentum=cifar['momentum'], weight_decay=cifar['weight_decay'])
        lr_lambda = lambda i: np.interp([i], [0, num_iter_warmup, num_iter_total], [0, 1, 0])[0]
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
    
    #transition=get_transition('dirichlet')
    transition=dirichlet_transition(device, num_classes, dirichlet['diagonal'], dirichlet['off_diagonal'], dirichlet['betas'])
    regularization, gamma = get_regularization(regularization_type,num_pairs,gamma)
    train_step = get_train_step(model, transition, optimizer, regularization, gamma)
    transition_matrix = trainset.actual_noise_rate
    test_accuracy,train_accuracy, tv = 0., 0.,1.

    test_acc_list=list()
    train_acc_list=list()
    for it, (x, y,_,_) in enumerate(take_cycle(num_iter_total + 1, train_loader)):
        # train
        train_step(x, y)
        
        scheduler.step()
        if it % num_iter_test == 0 or it == num_iter_total:
            # test
            t, z = predict_train(model, train_loader)
            train_accuracy = z.eq(t.argmax(dim=1)).sum().item() / len(z)
            t, z = predict_test(model, test_loader)
            test_accuracy = z.eq(t.argmax(dim=1)).sum().item() / len(z)
            train_acc_list.append(train_accuracy)
            test_acc_list.append(test_accuracy)
            # log
            if (transition.matrix) is not None:
                m=transition.matrix
                m = m.detach().cpu().numpy()
                tv = 100* 0.5 * np.abs(m - transition_matrix).sum(axis=1).mean()
                print(m,transition_matrix)
                kendaltau = kendall_tau(m,transition_matrix)
                print('Total variance:{} kendalltau: {}'.format(tv,kendaltau))

            print('iteration:{} train acc: {} test_acc: {}'.format(it, train_accuracy, test_accuracy))
            print('tv', tv, it)

    #final matrix
    if (transition.matrix) is not None:
        m=transition.matrix
        m = m.detach().cpu().numpy()
        print('matrix', m.tolist())
    filename = '{}_{}_{}.json'.format(args.dataset,args.noise_type,args.noise_rate)
    
    log_data = {'train_acc' : train_acc_list,
                    'test_acc' : test_acc_list, 'avg_total_variance':tv,'kendalltau':kendaltau}
    with open("./total_variance/log/"+filename, "w") as json_file:
            json.dump(log_data,json_file)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type = str, help = 'mnist, cifar10, or cifar100', default = 'mnist')
    parser.add_argument('--result_dir', type=str, help='dir to save result txt files', default='results')
    parser.add_argument('--noise_rate', type=float, help='corruption rate, should be less than 1', default=0.2)
    parser.add_argument('--noise_type', type=str, help='[pairflip, symmetric, asymmetric, asymmetric2, asymmetric3]', default='pairflip')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    train(args)