import numpy as np
import torch
from scipy.stats import kendalltau

def avg_total_variance(wm,tm):
    labels=wm.shape[0]
    temp=np.sum(abs(wm-tm),axis=1)
    return 1/labels*np.sum(temp/2)

def kendall_tau(wm,tm):
    labels=wm.shape[0]
    res=0
    for i in range(labels):
        tau, p_value=kendalltau(wm[i,:],tm[i,:])
        res+=tau
    return res/labels

def estimate_tm(test_loader,model,noise_layer,num_classes=10,device='cuda'):
    total=0
    res = torch.zeros((num_classes,num_classes)).to(device)
    model.eval()
    noise_layer.eval()
    with torch.no_grad():
        for image,target,_ in test_loader:
            h,logit = model(image.to(device),returnh=True)
            z = noise_layer(h) # [N x D x D]
            res += torch.sum(z,dim=0)
            total += z.shape[0]
    return (res/total).cpu().numpy()


def score_function(test_loader,model,noise_layer,true_transition,device='cuda'):
    res = list()
    model.eval()
    noise_layer.eval()
    true_transition = torch.tensor(true_transition).to(device)
    with torch.no_grad():
        for image,target,_ in test_loader:
            h,logit = model(image.to(device),returnh=True)
            z = noise_layer(h) # [N x D x D]
            score = kldivergence(z,true_transition)
            res += score
    return res


def confusion(train_loader,model,num_classes=10,device='cuda'):
    res = torch.zeros((num_classes,num_classes))
    model.eval()
    with torch.no_grad():
        for image,target,_,_ in train_loader:
            y = model(image.to(device))
            y = torch.argmax(y,dim=-1) 
            for y_i, t_i in zip(y,target):
                res[y_i,t_i]+=1
        for i in range(num_classes):
            sum_row = torch.sum(res[i,:])
            if sum_row>0:
                res[i,:]=res[i,:]/sum_row
        # norm = torch.sum(res,dim=-1).unsqueeze(-1)
        # res.div_(norm)
    return res


def kldivergence(z, true_transition):
    res = []
    for z_i in z:
        score = -true_transition*torch.log(z_i/(true_transition+1e-8)+1e-8)
        # print(score)
        score = score.sum().item()
        res.append(score)
    return res
    
def getth(score):
    score = np.asarray(score)
    th = np.median(score)
    clean_indicies = np.where(score<th)[0]
    ambiguous_indicies = np.where(score>=th)[0]
    return clean_indicies.tolist(),ambiguous_indicies.tolist()