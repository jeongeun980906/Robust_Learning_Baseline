# -*- coding:utf-8 -*-
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from model.cnn import MLPNet,CNN,CNN2
import torchvision.models as models
import numpy as np
from jocor.utils import accuracy

from .loss import loss_jocor,loss_mixup,get_index

import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn
class JoCoR:
    def __init__(self, args, train_dataset, device, input_channel, num_classes):

        # Hyper Parameters
        self.batch_size = 128
        self.args = args
        learning_rate = args.lr

        if args.forget_rate is None:
            if args.noise_type == "asymmetric":
                forget_rate = args.noise_rate / 2
            else:
                forget_rate = args.noise_rate
        else:
            forget_rate = args.forget_rate

        self.noise_or_not = train_dataset.noise_or_not

        # Adjust learning rate and betas for Adam Optimizer
        mom1 = 0.9
        mom2 = 0.1
        self.alpha_plan = [learning_rate] * args.n_epoch
        self.beta1_plan = [mom1] * args.n_epoch

        for i in range(args.epoch_decay_start, args.n_epoch):
            self.alpha_plan[i] = float(args.n_epoch - i) / (args.n_epoch - args.epoch_decay_start) * learning_rate
            self.beta1_plan[i] = mom2

        # define drop rate schedule
        self.rate_schedule = np.ones(args.n_epoch) * forget_rate
        self.rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate ** args.exponent, args.num_gradual)

        self.device = device
        self.num_iter_per_epoch = args.num_iter_per_epoch
        self.print_freq = args.print_freq
        self.co_lambda = args.co_lambda
        self.n_epoch = args.n_epoch
        self.train_dataset = train_dataset
        self.mixup = True
        self.num_classes = num_classes
        if args.model_type == "cnn":
            self.model1 = CNN(input_channel=input_channel, n_outputs=num_classes)
            self.model2 = CNN(input_channel=input_channel, n_outputs=num_classes)
        elif args.model_type == "cnn2":
            self.model1 = CNN2()
            self.model2 = CNN2()
        elif args.model_type == "mlp":
            self.model1 = MLPNet()
            self.model2 = MLPNet()
        elif args.model_type == 'resnet':
            self.model1 = models.resnet50(pretrained=True)
            self.model1.fc = nn.Linear(2048,num_classes)
            self.model2 = models.resnet50(pretrained=True)
            self.model2.fc = nn.Linear(2048,num_classes)
        self.model1.to(device)
        # print(self.model1.parameters)

        self.model2.to(device)
        # print(self.model2.parameters)
        if args.model_type == 'resnet':
            self.optimizer = torch.optim.SGD(list(self.model1.parameters()) + list(self.model2.parameters()),
                            lr=2e-3,momentum=0.9,weight_decay=1e-3)
        else:
            self.optimizer = torch.optim.Adam(list(self.model1.parameters()) + list(self.model2.parameters()),
                                          lr=learning_rate,weight_decay=1e-5)

        self.loss_fn = loss_jocor
        self.true_matrix = np.zeros((10,10))

        self.adjust_lr = args.adjust_lr

    # Evaluate the Model
    def evaluate(self, test_loader):
        print('Evaluating ...')
        self.model1.eval()  # Change model to 'eval' mode.
        self.model2.eval()  # Change model to 'eval' mode
        # confusion_matrix_1 = np.zeros((10,10))
        # confusion_matrix_2 = np.zeros((10,10))
        # true_matrix = np.zeros((10,10))

        correct1 = 0
        total1 = 0
        for images, labels, _ in test_loader:
            images = Variable(images).to(self.device)
            logits1 = self.model1(images)
            outputs1 = F.softmax(logits1, dim=1)
            _, pred1 = torch.max(outputs1.data, 1)
            total1 += labels.size(0)
            correct1 += (pred1.cpu() == labels).sum()
            # for i in range(labels.size(0)):
            #     confusion_matrix_1[pred1[i].item(), labels[i].item()] += 1

        correct2 = 0
        total2 = 0
        for images, labels, _ in test_loader:
            images = Variable(images).to(self.device)
            logits2 = self.model2(images)
            outputs2 = F.softmax(logits2, dim=1)
            _, pred2 = torch.max(outputs2.data, 1)
            total2 += labels.size(0)
            correct2 += (pred2.cpu() == labels).sum()
            # for i in range(labels.size(0)):
            #     confusion_matrix_2[pred2[i].item(), labels[i].item()] += 1
                #true_matrix[labels[i].item(),labels[i].item()] += 1

        acc1 = 100 * float(correct1) / float(total1)
        acc2 = 100 * float(correct2) / float(total2)

        # for i in range(10):
        #     if(np.sum(confusion_matrix_1[i,:]) != 0):
        #         confusion_matrix_1[i,:]= confusion_matrix_1[i,:]/np.sum(confusion_matrix_1[i,:])
        #     if(np.sum(confusion_matrix_2[i,:]) != 0):
        #         confusion_matrix_2[i,:]= confusion_matrix_2[i,:]/np.sum(confusion_matrix_2[i,:])
        #     true_matrix[i,:]= self.true_matrix[i,:]/np.sum(self.true_matrix[i,:])

        # filename = "./log/"+self.args.dataset + "_" + self.args.noise_type + "(" + str(self.args.noise_rate) +")/fig.png"
        # plt.clf()
        # fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(8*3,8))
        # fig.suptitle('{}_{}_{}'.format(self.args.dataset, self.args.noise_type, self.args.noise_rate), fontsize=20)
        # ax1.set_title('Confusion matrix1')
        # cm1_fig = sns.heatmap(confusion_matrix_1,ax=ax1,annot=True)
        # ax2.set_title('Confusion matrix2')
        # cm2_fig = sns.heatmap(confusion_matrix_2,ax=ax2,annot=True)
        # ax3.set_title('True')
        # true_matrix_fig = sns.heatmap(true_matrix,ax=ax3,annot=True)
        # fig.savefig(filename)

        return acc1, acc2

    # Train the Model
    def train(self, train_loader, epoch):
        print('Training ...')
        self.model1.train()  # Change model to 'train' mode.
        self.model2.train()  # Change model to 'train' mode

        if self.adjust_lr == 1:
            self.adjust_learning_rate(self.optimizer, epoch)

        train_total = 0
        train_correct = 0
        train_total2 = 0
        train_correct2 = 0
        pure_ratio_1_list = []
        pure_ratio_2_list = []

        for i, (images, labels, indexes, trues) in enumerate(train_loader):
            ind = indexes.cpu().numpy().transpose()
            if i > self.num_iter_per_epoch:
                break

            images = Variable(images).to(self.device)
            labels = Variable(labels).to(self.device)

            # Forward + Backward + Optimize
            logits1 = self.model1(images)
            prec1 = accuracy(logits1, labels, topk=(1,))
            train_total += 1
            train_correct += prec1

            logits2 = self.model2(images)
            prec2 = accuracy(logits2, labels, topk=(1,))
            train_total2 += 1
            train_correct2 += prec2
            if self.mixup:
                sampled_idx,pure_ratio_1,pure_ratio_2 = get_index(logits1, logits2, labels, self.rate_schedule[epoch],
                                                                 ind, self.noise_or_not, self.co_lambda)
                x = images[sampled_idx]
                y = labels[sampled_idx]
                y = torch.eye(self.num_classes)[y].to(self.device)
                loss_1,loss_2 = loss_mixup(x,y,self.model1,self.model2,self.co_lambda)
            else:      
                loss_1, loss_2, pure_ratio_1, pure_ratio_2 = self.loss_fn(logits1, logits2, labels, self.rate_schedule[epoch],
                                                                 ind, self.noise_or_not, self.co_lambda)

            self.optimizer.zero_grad()
            loss_1.backward()
            # torch.nn.utils.clip_grad_norm_(self.model1.parameters(), 1)
            # torch.nn.utils.clip_grad_norm_(self.model2.parameters(), 1)
            self.optimizer.step()

            pure_ratio_1_list.append(100 * pure_ratio_1)
            pure_ratio_2_list.append(100 * pure_ratio_2)

            # for i in range(labels.size(0)):
            #     self.true_matrix[labels[i].item(), trues[i].item()] += 1

            if (i + 1) % self.print_freq == 0:
                print(
                    'Epoch [%d/%d], Iter [%d/%d] Training Accuracy1: %.4F, Training Accuracy2: %.4f, Loss1: %.4f, Loss2: %.4f, Pure Ratio1 %.4f %% Pure Ratio2 %.4f %%'
                    % (epoch + 1, self.n_epoch, i + 1, len(self.train_dataset) // self.batch_size, prec1, prec2,
                       loss_1.data.item(), loss_2.data.item(), sum(pure_ratio_1_list) / len(pure_ratio_1_list), sum(pure_ratio_2_list) / len(pure_ratio_2_list)))

        train_acc1 = float(train_correct) / float(train_total)
        train_acc2 = float(train_correct2) / float(train_total2)
        return train_acc1, train_acc2, pure_ratio_1_list, pure_ratio_2_list

    def adjust_learning_rate(self, optimizer, epoch):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.alpha_plan[epoch]
            param_group['betas'] = (self.beta1_plan[epoch], 0.999)  # Only change beta1
