import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np
import json

from dividemix.loader import create_model,build_loader
from dividemix.solver import test,train,warmup,eval_train,eval_train_cloth,SemiLoss,NegEntropy
from dividemix.sdn_eval import eval

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--wd', '--weight decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--noise_mode',  default='sym')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--r', default=0.5, type=float, help='noise ratio')
parser.add_argument('--Adam', default=False,action='store_true',help='use adam optimizer')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--lr_step', default=150, type=int)
parser.add_argument('--lr_rate', default=0.1, type=float)
parser.add_argument('--data_path', default='./data/', type=str, help='path to dataset')
parser.add_argument('--dataset', default='mnist',choices = ['mnist','cifar10','cifar100','dirtymnist','dirtycifar','clothing1m'], type=str)
args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if args.dataset != 'clothing1m':
    filename = './dividemix/log/%s_%s_%.1f'%(args.dataset,args.noise_mode,args.r)
else:
    filename = './dividemix/log/%s'%(args.dataset)
stats_log=open(filename+'_stats.txt','w') 
test_log=open(filename+'_acc.txt','w')     

loader, warm_up, totalsize = build_loader(args,stats_log)
print('| Building net')
net1 = create_model(args)
net2 = create_model(args)
cudnn.benchmark = True

criterion = SemiLoss()

if args.Adam:
    optimizer1 = optim.Adam(net1.parameters(), lr=args.lr, weight_decay=args.wd,eps=1e-8)
    optimizer2 = optim.Adam(net2.parameters(), lr=args.lr, weight_decay=args.wd,eps=1e-8)
else:
    optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
if args.noise_mode=='asym' or args.dataset == 'clothing1m':
    conf_penalty = NegEntropy()
else:
    conf_penalty = None

all_loss = [[],[]] # save the history of losses from two networks

lr=args.lr
test_acc = []
for epoch in range(args.num_epochs+1):   
    if (epoch%args.lr_step) == 0 and epoch != 0:
        lr *= args.lr_rate      
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr       
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr  

    test_loader = loader.run('test')
    eval_loader = loader.run('eval_train')   
    
    if epoch<warm_up:       
        warmup_trainloader = loader.run('warmup')
        print('\nWarmup Net1')
        warmup(epoch,net1,optimizer1,warmup_trainloader,CEloss,conf_penalty,args)    
        print('\nWarmup Net2')
        warmup(epoch,net2,optimizer2,warmup_trainloader,CEloss,conf_penalty,args) 
   
    else:         
        if args.dataset == 'clothing1m':
            prob1,paths1 = eval_train_cloth(net1,eval_loader,CE,args)
            prob2,paths2 = eval_train_cloth(net1,eval_loader,CE,args)
            pred1 = (prob1 > args.p_threshold)  # divide dataset  
            pred2 = (prob2 > args.p_threshold)      
            
            print('\n\nTrain Net1')
            labeled_trainloader, unlabeled_trainloader = loader.run('train',pred2,prob2,paths=paths2) # co-divide
            train(epoch,net1,net2,optimizer1,labeled_trainloader, unlabeled_trainloader,criterion,args,warm_up)              # train net1
            print('\nTrain Net2')
            labeled_trainloader, unlabeled_trainloader = loader.run('train',pred1,prob1,paths=paths1) # co-divide
            train(epoch,net2,net1,optimizer2,labeled_trainloader, unlabeled_trainloader,criterion,args,warm_up)              # train net2
        
        else:
            prob1,all_loss[0]=eval_train(net1,eval_loader,all_loss[0],totalsize,CE,args)   
            prob2,all_loss[1]=eval_train(net2,eval_loader,all_loss[1],totalsize,CE,args)          
               
            pred1 = (prob1 > args.p_threshold)      
            pred2 = (prob2 > args.p_threshold)      
            
            print('\nTrain Net1')
            labeled_trainloader, unlabeled_trainloader = loader.run('train',pred2,prob2) # co-divide
            train(epoch,net1,net2,optimizer1,labeled_trainloader, unlabeled_trainloader,criterion,args,warm_up) # train net1  
            
            print('\nTrain Net2')
            labeled_trainloader, unlabeled_trainloader = loader.run('train',pred1,prob1) # co-divide
            train(epoch,net2,net1,optimizer2,labeled_trainloader, unlabeled_trainloader,criterion,args,warm_up) # train net2         

    acc = test(epoch,test_loader,net1,net2,test_log) 
    test_acc.append(acc) 

log_data = {
    'test_acc': test_acc
}
with open(filename + ".json", "w") as json_file:
        json.dump(log_data,json_file)

if args.dataset == 'dirtymnist' or args.dataset == 'dirtycifar':
    gt_amb_tm = loader.all_dataset.P
    gt_clean_tm = np.eye(10)
    eval(args,net1,net2,test_log,gt_amb_tm,gt_clean_tm)