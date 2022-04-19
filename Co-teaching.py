from __future__ import print_function 
import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from dataloader.cifar import CIFAR10, CIFAR100
from dataloader.mnist import MNIST
from model.cnn import CNN2,CNN
import torchvision.models as models

from dataloader.dirty_mnist import DirtyMNIST
from dataloader.dirty_cifar import dirtyCIFAR10
from dataloader.clothing1m import clothing1M
import torchvision.models as models

import argparse, sys
import numpy as np
import datetime
import json
import torch.nn as nn

from co_teaching.loss import loss_coteaching, loss_coteaching_plus

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type = float, default = 0.001)
parser.add_argument('--wd', type = float, default = 0.0)
parser.add_argument('--gpu', type = int, default = 0)
parser.add_argument('--result_dir', type = str, help = 'dir to save result txt files', default = 'results/')
parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.2)
parser.add_argument('--forget_rate', type = float, help = 'forget rate', default = None)
parser.add_argument('--noise_type', type = str, help='[pairflip, symmetric]', default='symmetric')
parser.add_argument('--num_gradual', type = int, default = 10, help='how many epochs for linear drop rate. This parameter is equal to Ek for lambda(E) in the paper.')
parser.add_argument('--dataset', type = str, help = 'mnist, cifar10, dirtymnist, dirtycifar10, or imagenet_tiny', default = 'mnist')
parser.add_argument('--n_epoch', type=int, default=200)
parser.add_argument('--optimizer', type = str, default='adam')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--epoch_decay_start', type=int, default=20)
parser.add_argument('--model_type', type = str, help='[coteaching, coteaching_plus]', default='coteaching_plus')
parser.add_argument('--fr_type', type = str, help='forget rate type', default='type_1')

args = parser.parse_args()

# Seed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(args.gpu)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu) 
torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode
torch.manual_seed(0)  # CPU seed
torch.cuda.manual_seed_all(0)  # GPU seed

# Hyper Parameters
batch_size = 128
learning_rate = args.lr 

# load dataset
if args.dataset=='mnist':
    input_channel=1
    num_classes=10
    args.top_bn = False
    args.epoch_decay_start = 5
    args.n_epoch = 20
    init_epoch = 1
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
    args.top_bn = False
    args.epoch_decay_start = 5
    args.n_epoch = 20
    init_epoch = 3
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

if args.dataset=='cifar10':
    input_channel=3
    num_classes=10
    args.top_bn = False
    args.epoch_decay_start = 80
    args.n_epoch = 200
    init_epoch = 10
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

if args.dataset=='cifar100':
    input_channel=3
    num_classes=100
    args.top_bn = False
    args.epoch_decay_start = 80
    args.n_epoch = 200
    init_epoch = 10
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

if args.dataset=='dirtycifar10':
    input_channel=3
    num_classes=10
    args.top_bn = False
    args.epoch_decay_start = 80
    args.n_epoch = 200
    init_epoch = 10
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
if args.dataset == 'clothing1m':
    input_channel=3
    num_classes=14
    args.top_bn = False
    args.epoch_decay_start = 40
    args.n_epoch = 80
    init_epoch = 2
    batch_size = 32

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
    forget_rate=args.noise_rate
else:
    forget_rate=args.forget_rate

noise_or_not = train_dataset.noise_or_not

# Adjust learning rate and betas for Adam Optimizer
mom1 = 0.9
mom2 = 0.1
alpha_plan = [learning_rate] * args.n_epoch
beta1_plan = [mom1] * args.n_epoch
for i in range(args.epoch_decay_start, args.n_epoch):
    alpha_plan[i] = float(args.n_epoch - i) / (args.n_epoch - args.epoch_decay_start) * learning_rate
    beta1_plan[i] = mom2

def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr']=alpha_plan[epoch]
        param_group['betas']=(beta1_plan[epoch], 0.999) 
       
# define drop rate schedule
def gen_forget_rate(fr_type='type_1'):
    if fr_type=='type_1':
        rate_schedule = np.ones(args.n_epoch)*forget_rate
        rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate, args.num_gradual)

    #if fr_type=='type_2':
    #    rate_schedule = np.ones(args.n_epoch)*forget_rate
    #    rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate, args.num_gradual) 
    #    rate_schedule[args.num_gradual:] = np.linspace(forget_rate, 2*forget_rate, args.n_epoch-args.num_gradual)
        
    return rate_schedule

rate_schedule = gen_forget_rate(args.fr_type)
  
save_dir = args.result_dir +'/' +args.dataset+'/%s/' % args.model_type

if not os.path.exists(save_dir):
    os.system('mkdir -p %s' % save_dir)

model_str = args.dataset + '_%s_' % args.model_type + args.noise_type + '_' + str(args.noise_rate)

txtfile = save_dir + "/" + model_str + ".txt"
nowTime=datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
if os.path.exists(txtfile):
    os.system('mv %s %s' % (txtfile, txtfile+".bak-%s" % nowTime))


def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# Train the Model
def train(train_loader,epoch, model1, optimizer1, model2, optimizer2):
    print('Training %s...' % model_str)
    
    train_total=0
    train_correct=0 
    train_total2=0
    train_correct2=0 

    for i, (data, labels, indexes,_) in enumerate(train_loader):
        ind=indexes.cpu().numpy().transpose()
        labels = Variable(labels).cuda()
        # print(labels)
        data = Variable(data).cuda()
        # Forward + Backward + Optimize
        logits1=model1(data)
        prec1,  = accuracy(logits1, labels, topk=(1, ))
        train_total+=1
        train_correct+=prec1

        logits2 = model2(data)
        prec2,  = accuracy(logits2, labels, topk=(1, ))
        train_total2+=1
        train_correct2+=prec2
        if epoch < init_epoch:
            loss_1, loss_2, _, _ = loss_coteaching(logits1, logits2, labels, rate_schedule[epoch], ind, noise_or_not)
        else:
            if args.model_type=='coteaching_plus':
                loss_1, loss_2, _, _ = loss_coteaching_plus(logits1, logits2, labels, rate_schedule[epoch], ind, noise_or_not, epoch*i)
            else:
                loss_1, loss_2, _, _ = loss_coteaching(logits1, logits2, labels, rate_schedule[epoch], ind, noise_or_not)
        # print(loss_1)
        optimizer1.zero_grad()
        loss_1.backward()
        # torch.nn.utils.clip_grad_norm_(model1.parameters(), 1)
        optimizer1.step()
        optimizer2.zero_grad()
        loss_2.backward()
        # torch.nn.utils.clip_grad_norm_(model2.parameters(), 1)
        optimizer2.step()
        # print('Epoch [%d/%d], Iter [%d/%d] Training Accuracy1: %.4F, Training Accuracy2: %.4f, Loss1: %.4f, Loss2: %.4f' 
        #           %(epoch+1, args.n_epoch, i+1, len(train_dataset)//batch_size, prec1, prec2, loss_1.item(), loss_2.item()))

    train_acc1=float(train_correct)/float(train_total)
    train_acc2=float(train_correct2)/float(train_total2)
    return train_acc1, train_acc2

# Evaluate the Model
def evaluate(test_loader, model1, model2):
    print('Evaluating %s...' % model_str)
    model1.eval()    # Change model to 'eval' mode.
    correct1 = 0
    total1 = 0
    for data, labels, _ in test_loader:
        # print(data)
        data = Variable(data).cuda()
        logits1 = model1(data)
        outputs1 = F.softmax(logits1, dim=1)
        _, pred1 = torch.max(outputs1.data, 1)
        total1 += labels.size(0)
        correct1 += (pred1.cpu() == labels.long()).sum()

    model2.eval()    # Change model to 'eval' mode 
    correct2 = 0
    total2 = 0
    for data, labels, _ in test_loader:
        data = Variable(data).cuda()
        logits2 = model2(data)
        outputs2 = F.softmax(logits2, dim=1)
        _, pred2 = torch.max(outputs2.data, 1)
        total2 += labels.size(0)
        correct2 += (pred2.cpu() == labels.long()).sum()
 
    acc1 = 100*float(correct1)/float(total1)
    acc2 = 100*float(correct2)/float(total2)
    return acc1, acc2

def main():
    # Data Loader (Input Pipeline)
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
    if args.dataset == 'mnist' or args.dataset == 'dirtymnist':
        clf1 = CNN2()
        optimizer1 = torch.optim.Adam(clf1.parameters(), lr=learning_rate,weight_decay=args.wd)
    if args.dataset == 'cifar10' or args.dataset == 'dirtycifar10' or args.dataset == 'cifar100':
        clf1 = CNN(num_classes)
        optimizer1 = torch.optim.Adam(clf1.parameters(), lr=learning_rate,weight_decay=args.wd)
    else:
        clf1 = models.resnet50(pretrained=True)
        clf1.fc = nn.Linear(2048,num_classes)
        optimizer1 = torch.optim.SGD(clf1.parameters(), lr=learning_rate,weight_decay=1e-3,momentum=0.9)
    clf1.cuda()
    
    
    if args.dataset == 'mnist' or args.dataset == 'dirtymnist':
        clf2 = CNN2()
        optimizer2 = torch.optim.Adam(clf2.parameters(), lr=learning_rate,weight_decay=args.wd)
    if args.dataset == 'cifar10' or args.dataset == 'dirtycifar10' or args.dataset == 'cifar100':
        clf2 = CNN(num_classes)
        optimizer2 = torch.optim.Adam(clf2.parameters(), lr=learning_rate,weight_decay=args.wd)
    else:
        clf2 = models.resnet50(pretrained=True)
        clf2.fc = nn.Linear(2048,num_classes)
        optimizer2 = torch.optim.SGD(clf2.parameters(), lr=learning_rate,weight_decay=1e-3,momentum=0.9)

    clf2.cuda()

    epoch=0
    train_acc1=0
    train_acc2=0
    train_acc1_list =list()
    train_acc2_list =list()
    test_acc1_list =list()
    test_acc2_list =list()
    # evaluate models with random weights
    test_acc1, test_acc2=evaluate(test_loader, clf1, clf2)
    print('Epoch [%d/%d] Test Accuracy on the %s test data: Model1 %.4f %% Model2 %.4f %%' % (epoch+1, args.n_epoch, len(test_dataset), test_acc1, test_acc2))
    # save results
    # training
    for epoch in range(1, args.n_epoch):
        # train models
        clf1.train()
        clf2.train()

        adjust_learning_rate(optimizer1, epoch)
        adjust_learning_rate(optimizer2, epoch)

        train_acc1, train_acc2 = train(train_loader, epoch, clf1, optimizer1, clf2, optimizer2)
        # evaluate models
        test_acc1, test_acc2 = evaluate(test_loader, clf1, clf2)
        train_acc1_list.append(train_acc1)
        train_acc2_list.append(train_acc2)
        test_acc1_list.append(test_acc1)
        test_acc2_list.append(test_acc2)
        # save results
        print('Epoch [%d/%d] Test Accuracy on the %s test data: Model1 %.4f %% Model2 %.4f %%' % (epoch+1, args.n_epoch, len(test_dataset), test_acc1, test_acc2))
    
    plus= 1 if args.model_type=='coteaching_plus' else 0
    if args.dataset != 'clothing1m':
        filename = '{}_{}_{}_{}.json'.format(args.dataset,args.noise_type,args.noise_rate,plus)
    else:
        filename = '{}_{}.json'.format(args.dataset,plus)
    log_data = {'train_acc1' : train_acc1_list,
                    'train_acc2' : train_acc2_list,
                    'test_acc1' : test_acc1_list,
                    'test_acc2' : test_acc2_list}
    with open("./co_teaching/log/"+filename, "w") as json_file:
        json.dump(log_data,json_file)

if __name__=='__main__':
    main()
