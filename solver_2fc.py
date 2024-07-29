import pickle

import torch
import torch.nn as nn
import os
from torch import optim
# from model_optical_matrix import VisionTransformer
from model_2fc import *
from sklearn.metrics import confusion_matrix, accuracy_score
from data_loader import get_loader
import time

import numpy as np
class Solver(object):
    def __init__(self, args):
        self.args = args

        self.train_loader, self.test_loader = get_loader(args)

        self.model = cnn_fc(args).cuda()
        self.ce = nn.CrossEntropyLoss()

        print('--------Network--------')
        print(self.model)


        # self.model.load_state_dict(torch.load('model/model_2fc.pth'))

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
                class_out,x_cnn,x_fc  = self.model(imgs,i)
            _, predicted = torch.max(class_out.data, 1)

            actual += labels.tolist()
            pred += predicted.tolist()
            # if i >10:
            #     break
            # i = i+1
        acc = accuracy_score(y_true=actual, y_pred=pred) * 100
        cm = confusion_matrix(y_true=actual, y_pred=pred, labels=range(self.args.n_classes))

        return acc, cm

    def test(self):
        train_acc, cm = self.test_dataset('train')
        print("Tr Acc: %.2f" % train_acc)
        print(cm)

        test_acc, cm = self.test_dataset('test')
        print("Te Acc: %.2f" % test_acc)
        print(cm)

        return train_acc, test_acc

    def train(self):
        iter_per_epoch = len(self.train_loader)

        optimizer = optim.Adam(self.model.parameters(), self.args.lr, weight_decay=1e-3)
        cos_decay = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.6)
        loss_during_train = []
        accuracy__during_train = []
        for epoch in range(self.args.epochs):

            self.model.train()
            start = time.time()
            for i, (imgs, labels) in enumerate(self.train_loader):


                imgs, labels = imgs.cuda(), labels.cuda()
                imgs.require_grad = True
                logits,x_cnn,x_fc  = self.model(imgs,i)

                clf_loss = self.ce(logits, labels)

                optimizer.zero_grad()
                clf_loss.backward()
                # 4 5 6 7 8 9对应qkv的weight和bias梯度,xq = w * x_1 + bias


                optimizer.step()


                if i % 500 == 0 or i == (iter_per_epoch - 1):
                    print('Ep: %d/%d, it: %d/%d, err: %.4f' % (epoch + 1, self.args.epochs, i + 1, iter_per_epoch, clf_loss))
                    loss_during_train.append(clf_loss)
            end = time.time()
            print(start - end)
            if epoch%10==0:
                test_acc, cm = self.test_dataset('train')
                accuracy__during_train.append(test_acc)
                print("Test acc: %0.2f" % test_acc)
                print(cm, "\n")

            cos_decay.step()
        folder_name = 'model'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        model_name = 'model_2fc.pth'
        with open(os.path.join(folder_name, 'loss.pkl'), 'wb') as f:
            # 将列表数据转换为字符串并写入文件
            pickle.dump(loss_during_train, f)
        with open(os.path.join(folder_name, 'acc.pkl'), 'wb') as f:
            # 将列表数据转换为字符串并写入文件
            pickle.dump(accuracy__during_train, f)
        torch.save(self.model.state_dict(), os.path.join(folder_name, model_name))
