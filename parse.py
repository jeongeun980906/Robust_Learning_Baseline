import json
from os import close
import matplotlib.pyplot as plt
import numpy as np

DATA = 'cifar100'
LEN = 10
train_res=dict()
test_res=dict()

NOISE=['symmetric_0.2','symmetric_0.5','symmetric_0.8','asymmetric_0.4']

def parse2(filename):
    with open(filename, "r") as json_file:
        data=json.load(json_file)
        temp = data['test_acc']
        temp1 = data['AUROC']
        temp2 = data['AVT_amb']*100
        temp3 = data['KDT_amb']
        temp4 = data['AVT_clean']*100
        temp5 = data['KDT_clean']
        last = temp[-LEN:]
        last = np.asarray(last)*100
        mean = np.mean(last)
        std = np.sum(np.abs(last-mean),axis=0)/last.shape[0]
        print(n,noise,round(mean,2),'$\pm$',round(std,2))
        print(n,noise,round(temp4,2),'&',round(temp5,4))
        print(n,noise,round(temp2,2),'&',round(temp3,4))
        print(n,noise,temp1)

def parse3(filename):
    with open(filename, "r") as json_file:
        data=json.load(json_file)
        temp = data['test_acc']
        temp2 = data['AVT']*100
        temp3 = data['KDT']
        last = temp[-LEN:]
        last = np.asarray(last)*100
        mean = np.mean(last)
        std = np.sum(np.abs(last-mean),axis=0)/last.shape[0]
        print(n,noise,round(mean,2),'$\pm$',round(std,2))
        print(n,noise,round(temp2,2),'&',round(temp3,4))

for noise in NOISE:
    n = 'noise_adaptation'
    filename = "./{}/log/{}_{}.json".format(n,DATA,noise)
    parse3(filename)