import torch
import numpy as np
from scipy.stats import kendalltau
from dividemix.loader import noisy_test_loader,sdn_loader
from sklearn.metrics import precision_recall_curve,roc_auc_score

def af(test_loader,net1,net2,device='cuda'):
    res = []
    net1.eval()
    net2.eval()
    with torch.no_grad():
        for batch_in,_ in test_loader:
            pred1 = net1(batch_in.to(device))
            pred2 = net2(batch_in.to(device))
            pred = (torch.softmax(pred1,dim=-1)+torch.softmax(pred2,dim=-1))/2
            entropy = torch.sum(-pred*torch.log(pred+1e-8),dim=-1)
            # print(entropy)
            res += entropy.cpu().numpy().tolist()
    return res

def get_th(res):
    res = np.asarray(res)
    th = np.median(res)
    index_clean = np.where(res<th)[0]
    index_amb = np.where(res>=th)[0]
    return index_clean.tolist(), index_amb.tolist()

def transition_matrix(test_loader,net1,net2,device='cuda'):
    res = torch.eye(10)
    net1.eval()
    net2.eval()
    with torch.no_grad():
        for batch_in,_ in test_loader:
            pred1 = net1(batch_in.to(device))
            pred2 = net2(batch_in.to(device))
            pred = (torch.softmax(pred1,dim=-1)+torch.softmax(pred2,dim=-1))/2
            _, target = torch.max(pred,1)
            for p,t in zip(pred,target):
                res[t,:] += p.cpu().numpy()
    norm = torch.sum(res,dim=-1).unsqueeze(-1)
    res.div_(norm)
    return res.numpy()

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


def eval(args,net1,net2,test_log,gt_amb_tm,gt_clean_tm):
    test_noisy_loader,length = noisy_test_loader(args)
    entropy = af(test_noisy_loader,net1,net2,device='cuda')
    clean_index,amb_index = get_th(entropy)

    gt = [0]*length+[1]*length
    AUROC = roc_auc_score(gt,entropy)
    test_log.write("AUROC: %.4f\n"%(AUROC))
    test_log.flush()

    del test_noisy_loader
    
    clean_loader,amb_loader= sdn_loader(args,clean_index,amb_index)
    clean_tm = transition_matrix(clean_loader,net1,net2)
    amb_tm = transition_matrix(amb_loader,net1,net2)

    clean_KTD = kendall_tau(gt_clean_tm,clean_tm)
    amb_KTD = kendall_tau(gt_amb_tm,amb_tm)

    clean_ATV = avg_total_variance(gt_clean_tm,clean_tm)
    amb_ATV= avg_total_variance(gt_amb_tm,amb_tm)

    test_log.write("CLEAN AVT: [%.3f] KTD: [%.4f]\n"%(clean_ATV*100,clean_KTD))
    test_log.flush()
    test_log.write("AMB AVT: [%.3f] KTD: [%.4f]"%(amb_ATV*100,amb_KTD))
    test_log.flush()
