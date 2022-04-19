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

class dirtyCIFAR10(Dataset): 
    def __init__(self, noise_rate, noise_type, root, transform, train, index=None,test_noisy=False, log=''): 
        
        self.r = noise_rate # noise ratio
        self.transform = transform
        self.train = train
        self.transition = {0:0,2:0,4:7,7:7,1:1,9:1,3:5,5:3,6:6,8:8} # class transition for asymmetric noise
        root = os.path.join(root,'cifar-10-batches-py')
        if self.train == False:         
            test_dic = unpickle('%s/test_batch'%root)            
            if test_noisy:
                test_data = test_dic['data']
                test_data = test_data.reshape((10000, 3, 32, 32))
                test_data = test_data.transpose((0, 2, 3, 1))
                test_label = test_dic['labels']
                
                clean_test_data = test_data[:5000]
                noisy_test_data  = test_data[5000:]
                clean_test_label = test_label[:5000]
                noisy_test_label = test_label[5000:]

                noisy_test_data = self.cutmix(noisy_test_data,noisy_test_label,0.5)
                self.test_data = np.concatenate((clean_test_data,noisy_test_data),axis=0)
                self.test_label = test_label
                if index != None:
                    index = np.asarray(index)
                    print(index.dtype)
                    self.test_data = self.test_data[index.astype(int)]
                    self.test_label = np.asarray(self.test_label)[index.astype(int)]
                    self.test_label = self.test_label.tolist()
            else:
                self.test_data = test_dic['data']
                self.test_label = test_dic['labels']  
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))  
        else:    
            train_data=[]
            train_label=[]
            for n in range(1,6):
                dpath = '%s/data_batch_%d'%(root,n)
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
            
            if noise_type == 'symmetric':
                noisy_train_label=np.asarray([[noisy_train_label[i]] for i in range(len(noisy_train_label))])
                noisy_train_label, self.actual_noise_rate = noisify_multiclass_symmetric(noisy_train_label, self.r, random_state=0, nb_classes=10)
                noisy_train_label=[i[0] for i in noisy_train_label]
                noise_label = clean_train_label + noisy_train_label
            elif noise_type == 'asymmetric':
                noisy_train_label=np.asarray([[noisy_train_label[i]] for i in range(len(noisy_train_label))])
                noisy_train_label, self.actual_noise_rate = noisify_cifar10_asymmetric(noisy_train_label, self.r, random_state=0)
                noisy_train_label=[i[0] for i in noisy_train_label]
                noise_label = clean_train_label + noisy_train_label
                # noise_label = clean_train_label
                # idx = list(range(25000))
                # random.shuffle(idx)
                # num_noise = int(self.r*25000)            
                # noise_idx = idx[:num_noise]
                # for i in range(25000):
                #     if i in noise_idx:  
                #         noiselabel = self.transition[noisy_train_label[i]]
                #         noise_label.append(noiselabel)                    
                #     else:    
                #         noise_label.append(noisy_train_label[i])  
                # print(len(noise_label))
    

            noisy_train_data = self.cutmix(noisy_train_data,noisy_train_label,0.5)
            train_data = np.concatenate((clean_train_data,noisy_train_data),axis=0)
            self.noise_or_not = np.transpose(noise_label)==np.transpose(train_label)
            self.train_data = train_data
            self.noise_label = noise_label
            self.train_label = train_label
            
      
    
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
        if self.train:
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img = self.transform(img) 
            true = self.train_label[index]
            return img,target,index,true
        else:   
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img) 
            return img, target, index        
        
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)         

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
