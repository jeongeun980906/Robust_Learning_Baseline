from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import os
import torch
from torchnet.meter import AUCMeter
from numpy.testing import assert_array_almost_equal


class dirtymnist_dataset(Dataset): 
    def __init__(self, dataset, r, noise_mode, root_dir, transform, mode, noise_file='', 
                        pred=[], probability=[], log='',test_noisy=False,index=None): 
        
        self.r = r # noise ratio
        self.transform = transform
        self.mode = mode  
        # self.transition = {0:0,2:0,4:7,7:7,1:1,9:1,3:5,5:3,6:6,8:8} # class transition for asymmetric noise
        self.processed_folder = 'processed'
        self.afolder = 'AmbiguousMNIST'
        self.training_file = 'training.pt'
        self.test_file = 'test.pt'
        resources = dict(
                    data=("amnist_samples.pt", "4f7865093b1d28e34019847fab917722"),
                    targets=("amnist_labels.pt", "3bfc055a9f91a76d8d493e8b898c3c95"),
                )
        self.resource_path = os.path.join(root_dir,'AmbiguousMNIST')
        if self.mode=='test':
            self.test_data, self.test_label = torch.load(
                os.path.join(root_dir, self.processed_folder, self.test_file)) 
            self.test_data = self.test_data/255
            self.test_data = self.test_data.reshape(-1, 1,28, 28)    
            self.test_data = self.test_data.sub_(0.1307).div_(0.3081)   
            if test_noisy:
                data_range = slice(60000,120000,6)
                test_noisy_data = torch.load(os.path.join(root_dir, self.afolder, resources['data'][0]))
                test_noisy_label = torch.load(os.path.join(root_dir, self.afolder, resources['targets'][0]))
                num_multi_labels = test_noisy_label.shape[1]
                test_noisy_data = test_noisy_data.expand(-1, num_multi_labels, 28, 28).reshape(-1, 1, 28, 28)
                test_noisy_label = test_noisy_label.reshape(-1)
                test_noisy_data = test_noisy_data[data_range]
                test_noisy_label = test_noisy_label[data_range]
                test_noisy_data = test_noisy_data.sub_(0.1307).div_(0.3081) 
                self.test_data = torch.cat((self.test_data,test_noisy_data),dim=0)
                self.test_label = torch.cat((self.test_label,test_noisy_label),dim=0)
                if index !=None:
                    self.test_data = self.test_data[index]
                    self.test_label = self.test_label[index]
                
        else:    
            train_clean_data, train_clean_label = torch.load(
                os.path.join(root_dir, self.processed_folder, self.training_file))
            train_clean_data = train_clean_data.float().div(255)
            train_noisy_data = torch.load(os.path.join(root_dir, self.afolder, resources['data'][0]))
            train_noisy_label = torch.load(os.path.join(root_dir, self.afolder, resources['targets'][0]))
            train_clean_data = train_clean_data.reshape(-1, 1, 28, 28)
            # Each sample has `num_multi_labels` many labels.
            num_multi_labels = train_noisy_label.shape[1]
            # print(train_noisy_data.mean(),train_noisy_data.std(),train_clean_data.mean())
            # Flatten the multi-label dataset into a single-label dataset with samples repeated x `num_multi_labels` many times
            train_noisy_data = train_noisy_data.expand(-1, num_multi_labels, 28, 28).reshape(-1, 1, 28, 28)
            train_noisy_label = train_noisy_label.reshape(-1)
            
            data_range = slice(None,60000)
            train_noisy_data = train_noisy_data[data_range]
            train_noisy_label = train_noisy_label[data_range]

            train_clean_label = train_clean_label.numpy().tolist()
            # print(len(train_data))
            if os.path.exists(noise_file):
                noise_label = json.load(open(noise_file,"r"))
                self.P = get_tm(noise_mode,self.r)
            else:    #inject noise  
                # print(train_noisy_label.shape)
                noise_label=np.asarray([[train_noisy_label[i]] for i in range(len(train_noisy_label))])
                if noise_mode == 'sym':
                    noise_label, self.P = noisify_multiclass_symmetric(noise_label, self.r, random_state=0, nb_classes=10) 
                elif noise_mode == 'asym':
                    noise_label, self.P = noisify_mnist_asymmetric(noise_label,self.r,random_state=0)
                noise_label=[int(i[0]) for i in noise_label]    
                noise_label = train_clean_label + noise_label
                # print(len(noise_label))
                print("save noisy labels to %s ..."%noise_file)        
                json.dump(noise_label,open(noise_file,"w"))  
                # print(train_clean_data.shape,train_noisy_data.shape)     
            train_label = train_clean_label+train_noisy_label.numpy().tolist()
            train_data = torch.cat((train_clean_data,train_noisy_data),dim=0)
            train_data = train_data.sub_(0.1307).div_(0.3081)
            # print(train_data.mean(),train_data.std())
            # print(train_data)
            if self.mode == 'all':
                self.train_data = train_data
                self.noise_label = noise_label
            else:                   
                if self.mode == "labeled":
                    pred_idx = pred.nonzero()[0]
                    self.probability = [probability[i] for i in pred_idx]   
                    
                    clean = (np.array(noise_label)==np.array(train_label))                                                       
                    auc_meter = AUCMeter()
                    auc_meter.reset()
                    auc_meter.add(probability,clean)        
                    auc,_,_ = auc_meter.value()               
                    log.write('Numer of labeled samples:%d   AUC:%.3f\n'%(pred.sum(),auc))
                    log.flush()      
                    
                elif self.mode == "unlabeled":
                    pred_idx = (1-pred).nonzero()[0]                                               
                
                self.train_data = train_data[pred_idx]
                self.noise_label = [noise_label[i] for i in pred_idx]                          
                print("%s data has a size of %d"%(self.mode,len(self.noise_label)))            
                
    def __getitem__(self, index):
        if self.mode=='labeled':
            img, target, prob = self.train_data[index], self.noise_label[index], self.probability[index]
            # img = Image.fromarray(img.numpy())
            # img1 = self.transform(img) 
            # img2 = self.transform(img) 
            return img, img, target, prob            
        elif self.mode=='unlabeled':
            img = self.train_data[index]
            # img = Image.fromarray(img.numpy())
            # img1 = self.transform(img) 
            # img2 = self.transform(img) 
            return img, img
        elif self.mode=='all':
            img, target = self.train_data[index], self.noise_label[index]
            # img = Image.fromarray(img.numpy())
            # img = self.transform(img)       
            return img, target, index        
        elif self.mode=='test':
            img, target = self.test_data[index], self.test_label[index]
            # img = Image.fromarray(img.numpy())
            # img = self.transform(img)            
            return img, target
           
    def __len__(self):
        if self.mode!='test':
            return len(self.train_data)
        else:
            return len(self.test_data)         
    
    
        
class dirtymnist_dataloader():  
    def __init__(self, dataset, r, noise_mode, batch_size, num_workers, root_dir, log, noise_file=''):
        self.dataset = dataset
        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.log = log
        self.noise_file = noise_file 
        self.transform_train = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize((0.1307), (0.3081)),
            ]) 
        self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize((0.1307), (0.3081))
            ])   
        self.all_dataset = dirtymnist_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="all",noise_file=self.noise_file)
        print(self.all_dataset.P)

    def run(self,mode,pred=[],prob=[]):
        if mode=='warmup':
            all_dataset =  self.all_dataset               
            trainloader = DataLoader(
                dataset= all_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)             
            return trainloader
                                     
        elif mode=='train':
            labeled_dataset = dirtymnist_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="labeled", noise_file=self.noise_file, pred=pred, probability=prob,log=self.log)              
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)   
            
            unlabeled_dataset = dirtymnist_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="unlabeled", noise_file=self.noise_file, pred=pred)                    
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)     
            return labeled_trainloader, unlabeled_trainloader
        
        elif mode=='test':
            test_dataset = dirtymnist_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='test')      
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return test_loader
        
        elif mode=='eval_train':
            eval_dataset = dirtymnist_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='all', noise_file=self.noise_file)      
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return eval_loader        


def get_tm(noise_type,noise_rate,nb_classes=10):
    if noise_type == 'sym':
        P = np.ones((nb_classes, nb_classes))
        n = noise_rate
        P = (n / (nb_classes - 1)) * P
        if n > 0.0:
            # 0 -> 1
            P[0, 0] = 1. - n
            for i in range(1, nb_classes-1):
                P[i, i] = 1. - n
            P[nb_classes-1, nb_classes-1] = 1. - n
    elif noise_type == 'asym':
        P = np.eye(nb_classes)
        n = noise_rate

        if n > 0.0:
            # 1 <- 7
            P[7, 7], P[7, 1] = 1. - n, n

            # 2 -> 7
            P[2, 2], P[2, 7] = 1. - n, n

            # 5 <-> 6
            P[5, 5], P[5, 6] = 1. - n, n
            P[6, 6], P[6, 5] = 1. - n, n

            # 3 -> 8
            P[3, 3], P[3, 8] = 1. - n, n
    return P

def noisify_multiclass_symmetric(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the symmetric way
    """
    P = np.ones((nb_classes, nb_classes))
    n = noise
    P = (n / (nb_classes - 1)) * P

    if n > 0.0:
        # 0 -> 1
        P[0, 0] = 1. - n
        for i in range(1, nb_classes-1):
            P[i, i] = 1. - n
        P[nb_classes-1, nb_classes-1] = 1. - n
        print(P)
        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    

    return y_train.tolist(), P

# basic function
def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
    # y = np.asarray(y)
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :][0], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y

def noisify_mnist_asymmetric(y_train, noise, random_state=None):
    """mistakes:
        1 <- 7
        2 -> 7
        3 -> 8
        5 <-> 6
    """
    nb_classes = 10
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # 1 <- 7
        P[7, 7], P[7, 1] = 1. - n, n

        # 2 -> 7
        P[2, 2], P[2, 7] = 1. - n, n

        # 5 <-> 6
        P[5, 5], P[5, 6] = 1. - n, n
        P[6, 6], P[6, 5] = 1. - n, n

        # 3 -> 8
        P[3, 3], P[3, 8] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy

    print(P)

    return y_train,P