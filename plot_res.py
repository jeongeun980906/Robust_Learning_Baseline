import json
from os import close
import matplotlib.pyplot as plt
import numpy as np

DATA = 'cifar10'
LEN = 10
train_res=dict()
test_res=dict()

NAME=['f_correction','co_teaching','jocor']
NOISE= ['symmetric_0.2','symmetric_0.5','symmetric_0.8','asymmetric_0.4','instance_0.2', 'instance_0.4'] 
NOISE2= ['0.2_symmetric','0.5_symmetric','0.8_symmetric','0.4_asymmetric','0.2_instance', '0.4_instance']

def parse4(filename):
    res = []
    with open(filename,'r')as f:
        while True:
            a = f.readline()
            if a == '' or len(a.split('AUROC')) != 1:
                break
            else:
                temp = a.split('Accuracy:')[1].split('\n')[0]
                res.append(float(temp))
    last = res[-LEN:]
    last = np.asarray(last)
    mean = np.mean(last)
    std = np.sum(np.abs(last-mean),axis=0)/last.shape[0]
    print(n,noise2,round(mean,2),'$\pm$',round(std,2))

def parse2(filename):
    with open(filename, "r") as json_file:
        data=json.load(json_file)
        # print(data)
        temp = data['test_acc2']
        last = temp[-LEN:]
        last = np.asarray(last)
        mean = np.mean(last)
        std = np.sum(np.abs(last-mean),axis=0)/last.shape[0]
        print(n,noise1,round(mean,2),'$\pm$',round(std,2))

def parse1(filename):
    with open(filename, "r") as json_file:
        data=json.load(json_file)
        temp = data['test_acc']
        last = temp[-LEN:]
        last = np.asarray(last)
        mean = np.mean(last)
        std = np.sum(np.abs(last-mean),axis=0)/last.shape[0]
        print(n,noise1,round(mean,2),'$\pm$',round(std,2))

def parse3(filename):
    with open(filename, "r") as json_file:
        data=json.load(json_file)
        temp = data['test_acc']
        last = temp[-LEN:]
        last = np.asarray(last)*100
        mean = np.mean(last)
        std = np.sum(np.abs(last-mean),axis=0)/last.shape[0]
        print(n,noise1,round(mean,2),'$\pm$',round(std,2))

for noise1,noise2 in zip(NOISE,NOISE2):
    # n = 'jocor'
    # filename = "./{}/log/{}_{}.json".format(n,DATA,noise1)
    # parse2(filename)
    # n = 'co_teaching'
    # filename = "./{}/log/{}_{}_{}.json".format(n,DATA,noise1,1)
    # parse2(filename)
    try:
        n = 'noise_adaptation'
        filename = "./{}/log/{}_{}.json".format(n,DATA,noise1)
        parse3(filename)
    except:
        pass
    try:    
        n = 'f_correction'
        filename = "./{}/log/{}_{}.json".format(n,DATA,noise1)
        parse1(filename)
    except:
        pass
    try:
        n = 'co_teaching'
        for i in range(2):
            filename = "./{}/log/{}_{}_{}.json".format(n,DATA,noise1,i)
            parse2(filename)
    except:
        pass
    try:
        n = 'jocor'
        filename = "./{}/log/{}_{}.json".format(n,DATA,noise1)
        parse2(filename)
    except:
        pass
    try:
        n = 'dividemix'
        filename = "./{}/log/{}_{}_1_acc.txt".format(n,DATA,noise2)
        parse4(filename)
    except:
        pass
# print(test_res)
# color=['lightcoral','peru','seagreen','royalblue','r']
# style=[':','-.',':','--','-']
# plt.figure(figsize=(14,10))
# plt.suptitle('MNIST Accuracy',fontsize=15)
# for i,noise in enumerate(NOISE):
#     plt.subplot(2,2,i+1)
#     plt.title(NOISE[i])
#     plt.xlabel("Epochs")
#     plt.ylabel("Accuracy")
#     for j,name in enumerate(NAME):
#         #print(train_res[j])
#         #plt.plot(train_res[name][noise],label=name+" train",color=color[j],lw=2,ls='--',marker='')
#         plt.plot(test_res[name][noise],label=name+' test',color=color[j],linestyle=style[j],lw=2,marker='')
#         Last=test_res[name][noise][-5:].copy()
#         Last=np.asarray(Last)
#         mean=np.mean(Last)
#         var=np.sum(abs(Last-mean))/Last.shape[0]
#         print("{} {} mean :{:.2f} var:{:.2f}".format(NAME[j],NOISE[i],mean,var))

#     if i==1:
#         plt.legend(bbox_to_anchor=(1.05, 1),loc=2, borderaxespad=0.)
# plt.tight_layout()
# plt.savefig('./res.jpg')