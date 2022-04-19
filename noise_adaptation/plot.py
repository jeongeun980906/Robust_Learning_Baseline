from sklearn.metrics import precision_recall_curve,roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import json

def plot_tm_ccn(args,out,true_noise,labels=10):
    
    DIR1='./noise_adaptation/log/{}_{}_{}_test_tm.png'.format(args.dataset,args.noise_type,args.noise_rate)
    DIR2='./noise_adaptation/log/{}_{}_{}_test.json'.format(args.dataset,args.noise_type,args.noise_rate)

    img3=np.round(out,2)
    img7=np.round(true_noise,2)
    NAME=args.dataset
    NAME=NAME.upper()
    NOISE_RATE=int(args.noise_rate*100)
    NOISE_TYPE= args.noise_type
    NOISE_TYPE=NOISE_TYPE.capitalize()
    
    fig, (ax3,ax4) = plt.subplots(2, 1, figsize=(6, 12))
    fig.suptitle('{} {} {}% Transition Matrix'.format(NAME,NOISE_TYPE,NOISE_RATE),fontsize=20)

    ax3.set_title('Predicted',fontsize=18)
    img = sns.heatmap(img3,ax=ax3,annot=True, cmap="YlGnBu", vmin=0, vmax=1)
    ax3.set_xlabel('Clean Label',fontsize=15)
    ax3.set_ylabel('Noise Label',fontsize=15)

    ax4.set_title('True',fontsize=18)
    img = sns.heatmap(img7,ax=ax4,annot=True, cmap="YlGnBu", vmin=0, vmax=1)
    ax4.set_xlabel('Clean Label',fontsize=15)
    ax4.set_ylabel('Noisy Label',fontsize=15)
    plt.tight_layout()
    fig.savefig(DIR1)

    save_log = {}
    save_log['TM']=img3.tolist()
    save_log['GT']=img7.tolist()
    with open(DIR2,'w') as json_file:
        json.dump(save_log,json_file,indent=4)

def plot_tm_sdn(args,out_clean,out_amb,true_clean,true_amb):
    img3=np.round(out_clean,2)
    img10=np.round(true_clean,2)
    img7 = np.round(out_amb,2)
    img9 = np.round(true_amb,2)
    NAME=args.dataset
    NAME=NAME.upper()
    NOISE_RATE=int(args.noise_rate*100)
    NOISE_TYPE= args.noise_type
    NOISE_TYPE=NOISE_TYPE.capitalize()
    
    DIR1='./noise_adaptation/log/{}_{}_{}_test_tm.png'.format(args.dataset,args.noise_type,args.noise_rate)
    DIR2='./noise_adaptation/log/{}_{}_{}_test.json'.format(args.dataset,args.noise_type,args.noise_rate)

    fig, ((ax3,ax7),(ax4,ax8)) = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle('Dirty {} {} {}% Transition Matrix \n'.format(NAME,NOISE_TYPE,NOISE_RATE),fontsize=20)
    ax3.set_title('Clean - Predicted',fontsize=18)
    img = sns.heatmap(img3,ax=ax3,annot=True, cmap="YlGnBu", vmin=0, vmax=1)
    ax3.set_xlabel('Clean Label',fontsize=15)
    ax3.set_ylabel('Noisy Label',fontsize=15)

    ax4.set_title('Clean - True',fontsize=18)
    img = sns.heatmap(img10,ax=ax4,annot=True, cmap="YlGnBu", vmin=0, vmax=1)
    ax4.set_xlabel('Clean Label',fontsize=15)
    ax4.set_ylabel('Noisy Label',fontsize=15)
    plt.tight_layout()

    ax7.set_title('Ambiguous - Predicted',fontsize=18)
    img = sns.heatmap(img7,ax=ax7,annot=True, cmap="YlGnBu", vmin=0, vmax=1)
    ax7.set_xlabel('Clean Label',fontsize=15)
    ax7.set_ylabel('Noisy Label',fontsize=15)

    ax8.set_title('Ambiguous - True',fontsize=18)
    img = sns.heatmap(img9,ax=ax8,annot=True, cmap="YlGnBu", vmin=0, vmax=1)
    ax8.set_xlabel('Clean Label',fontsize=15)
    ax8.set_ylabel('Noisy Label',fontsize=15)
    plt.tight_layout()
    fig.savefig(DIR1)