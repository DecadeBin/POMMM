import pickle

import torch
import torch.nn as nn
import os

from scipy.io import savemat
from torch import optim
# from model_optical_matrix import VisionTransformer

from model_fun import *
from sklearn.metrics import confusion_matrix, accuracy_score
from data_loader import get_loader
import time

import numpy as np
import random
def set_seed(seed):
    # 设置Python的随机种子
    random.seed(seed)
    # 设置NumPy的随机种子
    np.random.seed(seed)
    # 设置PyTorch的随机种子
    torch.manual_seed(seed)
    # 如果使用GPU，还需要设置CUDA的随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果你有多个GPU
        # 设置cuDNN的一些选项以确保结果的一致性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_seed(42)

class Solver(object):
    def __init__(self, args):
        self.args = args

        self.train_loader, self.test_loader = get_loader(args)
        if args.model_name == '2fc':
            self.model = cnn_fc(args).cuda()
        if args.model_name == 'vit':
            self.model = VisionTransformer(args).cuda()
        self.ce = nn.CrossEntropyLoss()

        print('--------Network--------')
        print(self.model)
        # if args.load_model==True:
        #     model_path = '../model/' + self.args.dset + '/' + self.args.model_name+'/'+self.args.dset+'_'+self.args.model_name+'_'+self.args.exp_type+'.pth'
        #     self.model.load_state_dict(torch.load(model_path))
        if args.load_model==True:
            test_exp_type =self.args.load_model_type #'omm'
            if self.args.hook==True or self.args.exp_type:
                model_path = '../model/' + self.args.dset + '/' + self.args.model_name + '/' + self.args.dset + '_' + self.args.model_name + '_' + test_exp_type + '_hook.pth'
            else:
                model_path = '../model/' + self.args.dset + '/' + self.args.model_name + '/' + self.args.dset + '_' + self.args.model_name + '_' + test_exp_type + '.pth'
            self.model.load_state_dict(torch.load(model_path,map_location=torch.device('cuda')))
    def test_dataset(self, db='test'):
        self.model.eval()

        actual = []
        pred = []

        if db.lower() == 'train':
            loader = self.train_loader
        else:
            loader = self.test_loader
        i = 0
        for (imgs, labels) in loader:
            imgs = imgs.cuda()

            with torch.no_grad():
                class_out = self.model(imgs, i)
            _, predicted = torch.max(class_out.data, 1)

            actual += labels.tolist()
            pred += predicted.tolist()
            # if i >500:
            #     break
            # i = i+1
        acc = accuracy_score(y_true=actual, y_pred=pred) * 100
        cm = confusion_matrix(y_true=actual, y_pred=pred, labels=range(self.args.n_classes))
        if self.args.load_model_type==self.args.exp_type:
            folder_name = '../model/' + self.args.dset + '/' + self.args.model_name+'/' +self.args.dset +'_'+self.args.model_name+'_'+self.args.exp_type+'_'
        else:
            folder_name = '../model/' + self.args.dset + '/' + self.args.model_name+'/' +self.args.dset +'_'+self.args.model_name+'_'+self.args.load_model_type+'2'+self.args.exp_type

        savemat(folder_name+db.lower()+'_'+str(self.args.N)+'_cm.mat',{'cm':cm})
        return acc, cm

    def test(self):
        train_acc, cm = self.test_dataset('train')
        print("Tr Acc: %.2f" % train_acc)
        print(cm)

        test_acc, cm = self.test_dataset('test')
        print("Te Acc: %.2f" % test_acc)
        print(cm)

        return train_acc, test_acc

    def train(self,args):
        iter_per_epoch = len(self.train_loader)

        optimizer = optim.Adam(self.model.parameters(), self.args.lr, weight_decay=1e-3)
        cos_decay = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.6)
        loss_during_train = []
        accuracy__during_train = []
        for epoch in range(self.args.epochs):

            start = time.time()

            self.model.train()
            for i, (imgs, labels) in enumerate(self.train_loader):

                # if args.exp_type=='three_stage':
                #     if epoch<=args.epochs/3:
                #         self.args.stage_time = 1
                #     if epoch > args.epochs / 3 and epoch < args.epochs*2 / 3:
                #         self.args.stage_time = 2
                #     if epoch > args.epochs *2/ 3:
                #         self.args.stage_time = 3
                if args.exp_type=='three_stage':
                    if epoch<=args.epochs/2:
                        self.args.stage_time = 2
                    if epoch > args.epochs /2:
                        self.args.stage_time = 3



                self.model.args = self.args
                imgs, labels = imgs.cuda(), labels.cuda()
                imgs.require_grad = True
                logits = self.model(imgs, i)

                clf_loss = self.ce(logits, labels)

                optimizer.zero_grad()
                clf_loss.backward()
                # 4 5 6 7 8 9对应qkv的weight和bias梯度,xq = w * x_1 + bias

                optimizer.step()
                # if i >50:
                #     break
                # i = i+1
                if i % 10 == 0 or i == (iter_per_epoch - 1):
                    print('Ep: %d/%d, it: %d/%d, err: %.4f' % (
                    epoch + 1, self.args.epochs, i + 1, iter_per_epoch, clf_loss))
                    loss_during_train.append(clf_loss)
                # break
            end = time.time()

            print(end - start)
            if epoch % 10 == 0 :
                self.model.args = self.args
                test_acc, cm = self.test_dataset('test')
                if epoch==self.args.epochs:
                    test_acc, cm = self.test_dataset('train')

                accuracy__during_train.append(test_acc)
                print("Test acc: %0.2f" % test_acc)
                print(cm, "\n")

            cos_decay.step()
            if (epoch+1)%10==0 or (epoch+1)==epoch:
                folder_name = '../model/' + self.args.dset + '/' + self.args.model_name

                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)
                if self.args.hook==True or self.args.exp_type=='three_stage':
                    model_name = self.args.dset + '_' + self.args.model_name + '_' + self.args.exp_type + '_hook.pth'
                else:
                    model_name = self.args.dset + '_' + self.args.model_name + '_' + self.args.exp_type + '_'+str(self.args.N)+'.pth'
                # with open(os.path.join(folder_name,
                #                        self.args.dset + '_' + self.args.model_name + '_' + self.args.exp_type + '_loss.pkl'),
                #           'wb') as f:
                #     # 将列表数据转换为字符串并写入文件
                #     pickle.dump(loss_during_train, f)
                # with open(os.path.join(folder_name,
                #                        self.args.dset + '_' + self.args.model_name + '_' + self.args.exp_type + '_acc.pkl'),
                #           'wb') as f:
                #     # 将列表数据转换为字符串并写入文件
                #     pickle.dump(accuracy__during_train, f)
                torch.save(self.model.state_dict(), os.path.join(folder_name, model_name))

