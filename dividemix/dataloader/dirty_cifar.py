from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import os
import torch
from torchnet.meter import AUCMeter      
from copy import deepcopy
from numpy.testing import assert_array_almost_equal

def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

class dirtycifar_dataset(Dataset): 
    def __init__(self, r, noise_mode, root_dir, transform, mode, noise_file='',
                     pred=[], probability=[], log='',test_noisy=False,index=None):  
        
        self.r = r # noise ratio
        self.transform = transform
        self.mode = mode  
        self.transition = {0:0,2:0,4:7,7:7,1:1,9:1,3:5,5:3,6:6,8:8} # class transition for asymmetric noise
        root_dir = os.path.join(root_dir,'cifar-10-batches-py') 
        if self.mode=='test':      
            test_dic = unpickle('%s/test_batch'%root_dir)
            test_data = test_dic['data']
            test_data = test_data.reshape((10000, 3, 32, 32))
            test_data = test_data.transpose((0, 2, 3, 1))  
            test_label = test_dic['labels']   
            if test_noisy:
                clean_test_data = test_data[:5000]  
                noisy_test_data = test_data[5000:]
                noisy_test_data = self.cutmix(noisy_test_data,test_label[5000:],0.5)
                self.test_data = np.concatenate((clean_test_data,noisy_test_data),axis=0)
                self.test_label = test_label
                if index !=None:
                    self.test_data = self.test_data[index]
                    self.test_label = np.asarray(self.test_label)[index].tolist()
            else:
                self.test_data = test_data
                self.test_label = test_label        
        else:    
            train_data=[]
            train_label=[]
            for n in range(1,6):
                dpath = os.path.join(root_dir,'data_batch_%d'%(n))
                data_dic = unpickle(dpath)
                train_data.append(data_dic['data'])
                train_label = train_label+data_dic['labels']
            train_data = np.concatenate(train_data)
            train_data = train_data.reshape((50000, 3, 32, 32))
            train_data = train_data.transpose((0, 2, 3, 1))
            
            clean_train_data = train_data[:25000]
            clean_train_label = train_label[:25000]
            noisy_train_data = train_data[25000:]
            noisy_train_label = train_label[25000:]
            if os.path.exists(noise_file):
                noise_label = json.load(open(noise_file,"r"))
                self.p = get_tm(noise_type,self.r)
            else:    #inject noise  
                if noise_mode == 'sym':
                    noisy_train_label=np.asarray([[noisy_train_label[i]] for i in range(len(noisy_train_label))])
                    noisy_train_label, self.P = noisify_multiclass_symmetric(noisy_train_label, self.r, random_state=0, nb_classes=10)
                    noisy_train_label=[i[0] for i in noisy_train_label]
                    noise_label = clean_train_label + noisy_train_label
                elif noise_mode == 'asym':
                    noisy_train_label=np.asarray([[noisy_train_label[i]] for i in range(len(noisy_train_label))])
                    noisy_train_label, self.P = noisify_cifar10_asymmetric(noisy_train_label, self.r, random_state=0)
                    noisy_train_label=[int(i[0]) for i in noisy_train_label]
                    noise_label = clean_train_label + noisy_train_label
                print("save noisy labels to %s ..."%noise_file)        
                json.dump(noise_label,open(noise_file,"w")) 
            noisy_train_data = self.cutmix(noisy_train_data,noisy_train_label,0.5)
            train_data = np.concatenate((clean_train_data,noisy_train_data),axis=0)
            # print(train_data.shape)
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
    
    def cutmix(self,data,labels,beta):
        # generate mixed sample
        labels=np.asarray(labels)
        mix_labels=[8,9,6,5,7,3,2,4,0,1]
        for j in range(10):
            indices_a = np.where(labels==j)[0]
            indices_b = np.where(labels==mix_labels[j])[0]
            rand_index = np.random.permutation(indices_b)
            length = indices_a.shape[0]
            shape2 = indices_b.shape[0]
            for i in range(length):
                lam = np.random.beta(beta, beta)
                bbx1, bby1, bbx2, bby2 = self.rand_bbox(data.shape, lam)
                data[indices_a[i], bbx1:bbx2, bby1:bby2, :] = deepcopy(data[rand_index[i%shape2], bbx1:bbx2, bby1:bby2,:])
        return data

    def rand_bbox(self,size, lam):
        W = size[1]
        H = size[2]
        cut_rat = np.sqrt(1. - lam)
        cut_rat= np.clip(cut_rat,0.3,0.7)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(int(cut_w/2),int(W-cut_w/2))
        cy = np.random.randint(int(cut_h/2),int(H-cut_h/2))

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2
    def __getitem__(self, index):
        if self.mode=='labeled':
            img, target, prob = self.train_data[index], self.noise_label[index], self.probability[index]
            img = Image.fromarray(img)
            img1 = self.transform(img) 
            img2 = self.transform(img) 
            return img1, img2, target, prob            
        elif self.mode=='unlabeled':
            img = self.train_data[index]
            img = Image.fromarray(img)
            img1 = self.transform(img) 
            img2 = self.transform(img) 
            return img1, img2
        elif self.mode=='all':
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target, index        
        elif self.mode=='test':
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target
           
    def __len__(self):
        if self.mode!='test':
            return len(self.train_data)
        else:
            return len(self.test_data)         

class dirtycifar_dataloader():  
    def __init__(self, r, noise_mode, batch_size, num_workers, root_dir, log, noise_file=''):
        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.log = log
        self.noise_file = noise_file
        self.transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
            ]) 
        self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
            ])    
        self.all_dataset = dirtycifar_dataset(noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="all",noise_file=self.noise_file)

    def run(self,mode,pred=[],prob=[]):
        if mode=='warmup':
            all_dataset = dirtycifar_dataset(noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="all",noise_file=self.noise_file)                
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)             
            return trainloader
                                     
        elif mode=='train':
            labeled_dataset = dirtycifar_dataset(noise_mode=self.noise_mode, 
                                    r=self.r, root_dir=self.root_dir, transform=self.transform_train, 
                                    mode="labeled", noise_file=self.noise_file, pred=pred, probability=prob,log=self.log)              
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)   
            
            unlabeled_dataset = dirtycifar_dataset(noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="unlabeled", noise_file=self.noise_file, pred=pred)                    
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)     
            return labeled_trainloader, unlabeled_trainloader
        
        elif mode=='test':
            test_dataset = dirtycifar_dataset(noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='test')      
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return test_loader
        
        elif mode=='eval_train':
            eval_dataset = dirtycifar_dataset(
                        noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, 
                        transform=self.transform_test, mode='all', noise_file=self.noise_file)      
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return eval_loader



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
        # print(P)
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


def noisify_cifar10_asymmetric(y_train, noise, random_state=None):
    """mistakes:
        automobile <- truck
        bird -> airplane
        cat <-> dog
        deer -> horse
    """
    nb_classes = 10
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # automobile <- truck
        P[9, 9], P[9, 1] = 1. - n, n

        # bird -> airplane
        P[2, 2], P[2, 0] = 1. - n, n

        # cat <-> dog
        P[3, 3], P[3, 5] = 1. - n, n
        P[5, 5], P[5, 3] = 1. - n, n

        # automobile -> truck
        P[4, 4], P[4, 7] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy

    return y_train, P


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
            # automobile <- truck
            P[9, 9], P[9, 1] = 1. - n, n

            # bird -> airplane
            P[2, 2], P[2, 0] = 1. - n, n

            # cat <-> dog
            P[3, 3], P[3, 5] = 1. - n, n
            P[5, 5], P[5, 3] = 1. - n, n

            # automobile -> truck
            P[4, 4], P[4, 7] = 1. - n, n
    return P
