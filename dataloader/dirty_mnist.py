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

class DirtyMNIST(Dataset): 
    def __init__(self, noise_rate, noise_type, root, train, test_noisy=False, 
                    index=None, log='',download=False): 
        
        self.r = noise_rate # noise ratio
        # self.transform = transform
        self.train = train  
        if download:
            self.download()
        # self.transition = {0:0,2:0,4:7,7:7,1:1,9:1,3:5,5:3,6:6,8:8} # class transition for asymmetric noise
        self.processed_folder = 'processed'
        self.afolder = 'AmbiguousMNIST'
        self.training_file = 'training.pt'
        self.test_file = 'test.pt'
        resources = dict(
                    data=("amnist_samples.pt", "4f7865093b1d28e34019847fab917722"),
                    targets=("amnist_labels.pt", "3bfc055a9f91a76d8d493e8b898c3c95"),
                )
        self.resource_path = os.path.join(root,'AmbiguousMNIST')
        if self.train == False:
            if test_noisy:
                test_clean_data, test_clean_label = torch.load(
                        os.path.join(root, self.processed_folder, self.test_file)) 
                test_clean_data = (test_clean_data/255).reshape(-1,1, 28, 28)  
                data_range = slice(60000,120000,6)
                test_noisy_data = torch.load(os.path.join(root, self.afolder, resources['data'][0]))
                test_noisy_label = torch.load(os.path.join(root, self.afolder, resources['targets'][0]))
                
                num_multi_labels = test_noisy_label.shape[1]
                # Flatten the multi-label dataset into a single-label dataset with samples repeated x `num_multi_labels` many times
                test_noisy_data = test_noisy_data.expand(-1, num_multi_labels, 28, 28).reshape(-1, 1, 28, 28)
                # print(test_noisy_data.shape)
                test_noisy_label = test_noisy_label.reshape(-1)
                test_noisy_data = test_noisy_data[data_range]
                test_noisy_label = test_noisy_label[data_range]
                # print(test_noisy_data.shape)
                self.test_data = torch.cat((test_clean_data,test_noisy_data),dim=0)
                self.test_label = torch.cat((test_clean_label,test_noisy_label),dim=0)
                self.test_data = self.test_data.sub_(0.1307).div_(0.3081)
                # print(self.test_data.shape,self.test_label.shape)
                if index != None:
                    index = torch.LongTensor(index)
                    self.test_data = self.test_data[index]
                    self.test_label = self.test_label[index]
                    print(self.test_data.shape)
            else:
                self.test_data, self.test_label = torch.load(
                    os.path.join(root, self.processed_folder, self.test_file)) 
                self.test_data = (self.test_data/255).reshape(-1,1, 28, 28)       
                self.test_data = self.test_data.sub_(0.1307).div_(0.3081)
                             
        else:    
            train_clean_data, train_clean_label = torch.load(
                os.path.join(root, self.processed_folder, self.training_file))
            train_clean_data = (train_clean_data/255).contiguous().reshape(-1, 1,28, 28)
            train_noisy_data = torch.load(os.path.join(root, self.afolder, resources['data'][0]))
            train_noisy_label = torch.load(os.path.join(root, self.afolder, resources['targets'][0]))
            # train_clean_data = train_clean_data.reshape(-1,1, 28, 28)
            # Each sample has `num_multi_labels` many labels.
            num_multi_labels = train_noisy_label.shape[1]
            # print(train_noisy_data.mean(),train_clean_data.mean())
            # Flatten the multi-label dataset into a single-label dataset with samples repeated x `num_multi_labels` many times
            train_noisy_data = train_noisy_data.expand(-1, num_multi_labels, 28, 28).reshape(-1, 1,28, 28)
            train_noisy_label = train_noisy_label.reshape(-1)
            
            data_range = slice(None,60000)
            train_noisy_data = train_noisy_data[data_range]
            train_noisy_label = train_noisy_label[data_range]

            train_clean_label = train_clean_label.numpy().tolist()
            
            # print(train_noisy_label.shape)
            noise_label=np.asarray([[train_noisy_label[i]] for i in range(len(train_noisy_label))])
            if noise_type == 'symmetric':
                noise_label, self.actual_noise_rate = noisify_multiclass_symmetric(noise_label, self.r, random_state=0, nb_classes=10) 
            elif noise_type == 'asymmetric':
                noise_label, self.actual_noise_rate = noisify_mnist_asymmetric(noise_label,self.r,random_state=0)
            noise_label=[int(i[0]) for i in noise_label]    
            noise_label = train_clean_label + noise_label
  
            train_label = train_clean_label+train_noisy_label.numpy().tolist()
            train_data = torch.cat((train_clean_data,train_noisy_data),dim=0)
            # print(train_data.shape)
            train_data = train_data.sub_(0.1307).div_(0.3081)
            print(train_data.mean(),train_data.std())
            print(train_data.shape,len(train_label),len(noise_label))

            self.noise_or_not = np.transpose(noise_label)==np.transpose(train_label)
            self.train_data = train_data
            self.noise_label = noise_label      
            self.train_label = train_label
            
                
    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.noise_label[index]
            # img = self.transform(img)     
            true = self.train_label[index]       
            return img, target, index ,true
        else:
            img, target = self.test_data[index], self.test_label[index]
            # img = self.transform(img) 
            return img, target, index  
           
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)         
        
    @property
    def data_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__)

    def resource_path(self, name):
        return os.path.join(self.data_folder, self.resources[name][0])

    def _check_exists(self) -> bool:
        return all(os.path.exists(self.resource_path(name)) for name in self.resources)

    def download(self) -> None:
        """Download the data if it doesn't exist in data_folder already."""

        if self._check_exists():
            return

        os.makedirs(self.data_folder, exist_ok=True)

        # download files
        for filename, md5 in self.resources.values():
            for mirror in self.mirrors:
                url = "{}{}".format(mirror, filename)
                try:
                    print("Downloading {}".format(url))
                    download_url(url, root=self.data_folder, filename=filename, md5=md5)
                except URLError as error:
                    print("Failed to download (trying next):\n{}".format(error))
                    continue
                except:
                    raise
                finally:
                    print()
                break
            else:
                raise RuntimeError("Error downloading {}".format(filename))

        print("Done!")

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