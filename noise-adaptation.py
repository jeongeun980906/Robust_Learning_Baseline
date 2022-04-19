import torch
import torch.optim as optim
import torch.nn as nn
import os
import argparse
import numpy as np
import json
import torch.nn as nn
from torchvision import datasets, transforms
from sklearn.metrics import precision_recall_curve,roc_auc_score

from dataloader.cifar import CIFAR10, CIFAR100
from dataloader.mnist import MNIST
from dataloader.dirty_mnist import DirtyMNIST
from dataloader.dirty_cifar import dirtyCIFAR10
from dataloader.clothing1m import clothing1M

from noise_adaptation.noise_layer import NoiseLayer
from noise_adaptation.solver import pretrain,test,train_adaptation,test_train
from noise_adaptation.eval import avg_total_variance,kendall_tau,confusion,estimate_tm,score_function,getth
from noise_adaptation.plot import plot_tm_ccn,plot_tm_sdn
from model.cnn import CNN,CNN2
import torchvision.models as models

# Training settings
parser = argparse.ArgumentParser(description='')
parser.add_argument('--gpu', type = int, default = 0)
parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for training, default: 128')
parser.add_argument('--epochs', type=int, default=20,
                    help='number of epochs to train, default: 10')
parser.add_argument('--warmup', type=int, default=1,
                    help='number of epochs to train, default: 10')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate, default: 0.1')
parser.add_argument('--lr_noise', type=float, default=1e-3,
                    help='initial learning rate, default: 0.1')
parser.add_argument('--wd_noise', type=float, default=0.01,
                    help='initial learning rate, default: 0.01')
parser.add_argument('--beta', type=float, default=0.8,
                    help='initial learning rate, default: 0.1')
parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist','cifar10','cifar100','dirtymnist','dirtycifar10','clothing1m'], help='dataset to train on, default: CIFAR10')
# parser.add_argument('--momentum', type=float, default=0.9,
#                     help='SGD momentum, default: 0.9')
parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.2)
parser.add_argument('--noise_type', type = str, help='[asymmetric, asymmetric]', default='symmetric')

args = parser.parse_args()

# Seed
# device = torch.device('cuda:{}'.format(str(args.gpu)))
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu) 
torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode
torch.manual_seed(0)  # CPU seed
torch.cuda.manual_seed_all(0)  # GPU seed
device = 'cuda'

# load dataset
if args.dataset=='mnist':
    input_channel=1
    num_classes=10
    args.epochs = 20
    h_dim = 64
    warmup = 1
    milestones = [10,15]
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
    input_channel=1
    num_classes=10
    args.epochs = 20
    warmup = 1
    h_dim = 64
    datalen = 10000
    milestones = [10,15]
    train_dataset = DirtyMNIST(root='./data/',
                                train=True, 
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                        )
    
    test_dataset = DirtyMNIST(root='./data/',
                            train=False, 
                            noise_type=args.noise_type,
                            noise_rate=args.noise_rate
                        )
    test_noisy_dataset = DirtyMNIST(root='./data/',
                            train=False, test_noisy=True,
                            noise_type=args.noise_type,
                            noise_rate=args.noise_rate
                        )

if args.dataset=='cifar10':
    input_channel=3
    num_classes=10
    args.epochs = 200
    warmup = 10
    h_dim = 256
    milestones = [100,150]
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
    input_channel=3
    num_classes=10
    args.epochs = 200
    warmup = 10
    h_dim = 256
    datalen = 5000
    milestones = [100,150]
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
                                noise_rate=args.noise_rate
                        )
    test_noisy_dataset = dirtyCIFAR10(root='./data/',
                                train=False, test_noisy=True,
                                transform=transform_test,
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                        )

if args.dataset=='cifar100':
    input_channel=3
    num_classes=100
    args.epochs = 200
    warmup = 10
    h_dim = 256
    milestones = [100,150]
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

    train_dataset = CIFAR100(root='./data/',
                                download=True,  
                                train=True, 
                                transform=transform_train,
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                        )
    test_dataset = CIFAR100(root='./data/',
                                download=True,  
                                train=False, 
                                transform=transform_test,
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                        )
if args.dataset == 'clothing1m':
    input_channel=3
    num_classes=14
    args.epochs = 80
    warmup = 2
    h_dim = 2048
    milestones = [30,60]

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
        num_samples = 1000*args.batch_size
    )
    test_dataset = clothing1M(
        root        = '/home/sungjoon.choi/seungyoun/Clothing1M',
        transform   = transform_val,
        mode        = 'test'
    )
if args.dataset == 'mnist' or args.dataset == 'dirtymnist':
    model = CNN2(n_outputs=num_classes).to(device)
    base_optimizer = optim.Adam(model.parameters(),lr=args.lr,weight_decay=1e-6,eps=1e-8)
if args.dataset == 'cifar10' or args.dataset == 'dirtycifar10' or args.dataset=='cifar100':
    model = CNN(n_outputs=num_classes).to(device)
    base_optimizer = optim.Adam(model.parameters(),lr=args.lr,weight_decay=1e-4,eps=1e-8)
else:
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(h_dim,num_classes)
    model = model.cuda()
    base_optimizer = optim.SGD(model.parameters(),lr=2e-3,weight_decay=1e-3,momentum=0.9)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
criterion = nn.CrossEntropyLoss()

scheduler = optim.lr_scheduler.MultiStepLR(base_optimizer, milestones=milestones, gamma=0.2)

test_acc_list=list()
train_acc_list=list()

for epoch in range(warmup):
    avg_loss = pretrain(train_loader,model,base_optimizer,criterion,device=device)
    train_acc = test_train(train_loader,model,device=device)
    test_acc = test(test_loader,model,device=device)
    print("Warm Up")
    print('Avg Loss: [%.3f]'%(avg_loss))
    print('EPOCH : [%d/%d] Train accuracy:[%.3f] test accuracy:[%.3f]'%(epoch, warmup, train_acc,test_acc))
    # train_acc_list.append(train_acc)
    # test_acc_list.append(test_acc)
    # scheduler.step()

theta = confusion(train_loader,model,num_classes=num_classes,device=device)
# print(theta)

# model.init_param()
noise_layer = NoiseLayer(torch.log(theta+1e-8),h_dim,num_classes).to(device)
noise_optimizer = optim.Adam(noise_layer.parameters(),lr=args.lr_noise,eps=1e-8)
# noise_scheduler = optim.lr_scheduler.MultiStepLR(noise_optimizer, milestones=milestones, gamma=0.5)

for epoch in range(warmup,args.epochs):
    avg_loss = train_adaptation(train_loader,model,noise_layer,base_optimizer,noise_optimizer,
                            criterion,device=device,weight_decay=args.wd_noise,beta=args.beta)
    test_acc = test(test_loader,model,device=device)
    train_acc = test_train(train_loader,model,device=device)
    print(args.dataset,args.noise_type,args.noise_rate)
    print('Avg Loss: [%.3f]'%(avg_loss))
    print('EPOCH : [%d/%d] Train accuracy:[%.3f] test accuracy:[%.3f]'%(epoch, args.epochs, train_acc,test_acc))
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    scheduler.step()
    # noise_scheduler.step()
if args.dataset == 'clothing1m':
    filename = '{}_{}_{}.json'.format(args.dataset,args.noise_type,args.noise_rate)
    log_data = {'train_acc' : train_acc_list,
                    'test_acc' : test_acc_list}

    with open("./noise_adaptation/log/"+filename, "w") as json_file:
            json.dump(log_data,json_file,indent=4)


elif args.dataset == 'mnist' or args.dataset == 'cifar10' or args.dataset == 'cifar100' or args.dataset == 'clothing1m':
    if args.noise_type != 'instance':
        transiton_matrix = estimate_tm(test_loader,model,noise_layer,num_classes=num_classes,device=device)
        true_transition = train_dataset.actual_noise_rate
        AVT = avg_total_variance(transiton_matrix,true_transition)
        KDT = kendall_tau(transiton_matrix,true_transition)
        print(AVT,KDT)
        # print(test_acc,transiton_matrix)
        plot_tm_ccn(args,transiton_matrix,true_transition)

        if args.dataset != 'clothing1m':
            filename = '{}_{}_{}.json'.format(args.dataset,args.noise_type,args.noise_rate)
        else:
            filename = '{}.json'.format(args.dataset)
        log_data = {'train_acc' : train_acc_list,
                    'test_acc' : test_acc_list,
                    'AVT': AVT, 'KDT':KDT}

        with open("./noise_adaptation/log/"+filename, "w") as json_file:
            json.dump(log_data,json_file,indent=4)
    else:
        filename = '{}_{}_{}.json'.format(args.dataset,args.noise_type,args.noise_rate)
        log_data = {'train_acc' : train_acc_list,
                    'test_acc' : test_acc_list}

        with open("./noise_adaptation/log/"+filename, "w") as json_file:
            json.dump(log_data,json_file,indent=4)
else:
    test_noisy_loader = torch.utils.data.DataLoader(test_noisy_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True)
    clean_true_transition = np.eye(num_classes)
    noisy_true_transition = train_dataset.actual_noise_rate
    
    score = score_function(test_noisy_loader,model,noise_layer,clean_true_transition,device=device)
    # print(score)
    gt = [0]*datalen+[1]*datalen
    AUROC = roc_auc_score(gt,score)
    print(AUROC)
    clean_indicies,ambiguous_indicies = getth(score)
    del test_noisy_loader,test_noisy_dataset
    if args.dataset == 'dirtymnist':
        test_clean_dataset = DirtyMNIST(root='./data/',
                                train=False, test_noisy=True, index = clean_indicies,
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                            )
        test_ambiguous_dataset = DirtyMNIST(root='./data/',
                                train=False, test_noisy=True, index = ambiguous_indicies,
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                            )
    elif args.dataset == 'dirtycifar10':
        test_clean_dataset = dirtyCIFAR10(root='./data/',
                                train=False, test_noisy=True, index = clean_indicies,
                                transform=transform_test,
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                            )
        test_ambiguous_dataset = dirtyCIFAR10(root='./data/',
                                train=False, test_noisy=True, index = ambiguous_indicies,
                                transform=transform_test,
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                            )
    test_clean_loader = torch.utils.data.DataLoader(test_clean_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True)
    test_ambiguous_loader = torch.utils.data.DataLoader(test_ambiguous_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True)
    transiton_clean_matrix = estimate_tm(test_clean_loader,model,noise_layer,num_classes=num_classes,device=device)
    transiton_ambiguous_matrix = estimate_tm(test_ambiguous_loader,model,noise_layer,num_classes=num_classes,device=device)
    
    AVT_c = avg_total_variance(transiton_clean_matrix,clean_true_transition)
    KDT_c = kendall_tau(transiton_clean_matrix,clean_true_transition)

    AVT_a = avg_total_variance(transiton_ambiguous_matrix,noisy_true_transition)
    KDT_a = kendall_tau(transiton_ambiguous_matrix,noisy_true_transition)
    
    plot_tm_sdn(args,transiton_clean_matrix,transiton_ambiguous_matrix,clean_true_transition,noisy_true_transition)
    filename = '{}_{}_{}.json'.format(args.dataset,args.noise_type,args.noise_rate)
    log_data = {'train_acc' : train_acc_list,
                'test_acc' : test_acc_list,'AUROC':AUROC,
                'AVT_clean': AVT_c, 'KDT_clean':KDT_c,
                'AVT_amb':AVT_a,'KDT_amb':KDT_a}

    with open("./noise_adaptation/log/"+filename, "w") as json_file:
        json.dump(log_data,json_file,indent=4)