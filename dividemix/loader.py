from model.cnn import CNN,CNN2
from dividemix.dataloader.mnist import mnist_dataloader
from dividemix.dataloader.cifar import cifar_dataloader
from dividemix.dataloader.dirty_mnist import dirtymnist_dataloader,dirtymnist_dataset
from dividemix.dataloader.dirty_cifar import dirtycifar_dataset,dirtycifar_dataloader
from dividemix.dataloader.clothing1m import clothing_dataloader

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn

def create_model(args):
    if args.dataset == 'cifar10' or args.dataset == 'cifar100' or args.dataset == 'dirtycifar':
        print('CNN')
        model = CNN(n_outputs = args.num_class)
    elif args.dataset == 'clothing1m':
        print('ResNet50')
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(2048,args.num_class)
    else:
        model = CNN2(n_outputs = args.num_class)
    model = model.cuda()
    return model

def build_loader(args,stats_log):
    if args.dataset=='cifar10':
        warm_up = 10
        totalsize = 50000
        loader = cifar_dataloader(args.dataset,r=args.r,noise_mode=args.noise_mode,batch_size=args.batch_size,num_workers=5,\
            root_dir=args.data_path,log=stats_log,noise_file='%s/%s_%.1f_%s.json'%(args.data_path,args.dataset,args.r,args.noise_mode))
    elif args.dataset=='cifar100':
        warm_up = 30
        args.num_classes = 100
        totalsize = 50000
        loader = cifar_dataloader(args.dataset,r=args.r,noise_mode=args.noise_mode,batch_size=args.batch_size,num_workers=5,\
            root_dir=args.data_path,log=stats_log,noise_file='%s/%s_%.1f_%s.json'%(args.data_path,args.dataset,args.r,args.noise_mode))
    elif args.dataset == 'dirtycifar':
        warm_up = 10
        totalsize = 50000
        loader = dirtycifar_dataloader(r=args.r,noise_mode=args.noise_mode,batch_size=args.batch_size,num_workers=5,\
            root_dir=args.data_path,log=stats_log,noise_file='%s/%s_%.1f_%s.json'%(args.data_path,args.dataset,args.r,args.noise_mode))
    
    elif args.dataset == 'mnist':
        warm_up = 2
        totalsize = 60000
        loader = mnist_dataloader(args.dataset,r=args.r,noise_mode=args.noise_mode,batch_size=args.batch_size,num_workers=5,\
            root_dir=args.data_path,log=stats_log,noise_file='%s/%s_%.1f_%s.json'%(args.data_path,args.dataset,args.r,args.noise_mode))

    elif args.dataset == 'dirtymnist':
        warm_up = 2
        totalsize = 120000
        loader = dirtymnist_dataloader(args.dataset,r=args.r,noise_mode=args.noise_mode,batch_size=args.batch_size,num_workers=5,\
                root_dir=args.data_path,log=stats_log,noise_file='%s/%s_%.1f_%s.json'%(args.data_path,args.dataset,args.r,args.noise_mode))
    elif args.dataset == 'clothing1m':
        warm_up = 1
        totalsize = args.batch_size*1000
        loader  = clothing_dataloader(root=args.data_path,batch_size=args.batch_size,num_workers=5,num_batches=1000)
    return loader,warm_up,totalsize

def noisy_test_loader(args):
    if args.dataset == 'dirtymnist':
        test_noisy_dataset = dirtymnist_dataset(args.dataset, args.r, args.noise_mode, args.data_path
                    , transforms.ToTensor(), 'test',test_noisy=True,index=None)
        test_noisy_loader = DataLoader(
                dataset=test_noisy_dataset, 
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=1)  
        length = 10000
    else:
        transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ])    
        test_noisy_dataset = dirtycifar_dataset(args.r, args.noise_mode, 
                        args.data_path,transform_test, 'test',test_noisy=True,index=None)
        test_noisy_loader = DataLoader(
                dataset=test_noisy_dataset, 
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=1)  
        length = 5000
    return test_noisy_loader,length

def sdn_loader(args,clean_index,amb_index):
    if args.dataset == 'dirtymnist':
        clean_dataset = dirtymnist_dataset(args.dataset, args.r, args.noise_mode, args.data_path
                    , transforms.ToTensor(), 'test',test_noisy=True,index=clean_index)
        clean_loader = DataLoader(
                    dataset=clean_dataset, 
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=1)  
        amb_dataset = dirtymnist_dataset(args.dataset, args.r, args.noise_mode, args.data_path
                        , transforms.ToTensor(), 'test',test_noisy=True,index=amb_index)
        amb_loader = DataLoader(
                    dataset=amb_dataset, 
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=1) 
    else:
        transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ]) 
        clean_dataset = dirtycifar_dataset(args.r, args.noise_mode, args.data_path,transform_test
                        , 'test',test_noisy=True,index=clean_index)
        clean_loader = DataLoader(
                    dataset=clean_dataset, 
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=1)  
        amb_dataset = dirtycifar_dataset(args.r, args.noise_mode, args.data_path,transform_test
                        ,'test',test_noisy=True,index=amb_index)
        amb_loader = DataLoader(
                    dataset=amb_dataset, 
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=1) 

    return clean_loader,amb_loader