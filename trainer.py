import model
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from math import log
import pandas as pd
import os, shutil

class Trainer:
    def __init__(self, trainData, testData, num_users=10):
        self.trainData = trainData
        self.testData = testData
        self.num_classes = num_users # set in main
        self.cuda_avail = torch.cuda.is_available()

        self.opt = {
            'batch_size': 32,
            'lambda': 0.95
        }

        self.create_model()

        if os.path.isdir("checkpoints"):
            shutil.rmtree("checkpoints")
            os.makedirs("checkpoints")
        else:
            os.makedirs("checkpoints")


    def loss(self, label_pred, forg_pred, labels, forgeries):
        out = 0.0
        for i in range(0, forgeries.shape[0]):
            temp = (forgeries[i] - 1)*(1 - self.opt['lambda'])
            temp2 = 0
            for j in range(0, self.num_classes):
                temp2 += labels[i][j]*log(label_pred[i][j])
            temp3 = forgeries[i]*log(forg_pred[i][0])
            temp4 = (1 - forgeries[i])*log(1-forg_pred[i][0])
            out += temp*temp2 - self.opt['lambda']*(temp3 + temp4)
        return out

    def create_model(self):
        self.model = model.Signet(self.num_classes)

        if self.cuda_avail:
            self.model.cuda()

        self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=1e-3, momentum=0.9, weight_decay=1e-4, nesterov=True)

    def save_models(self, epoch):
        torch.save(self.model.state_dict(), "checkpoints/Signet_{}.model".format(epoch))
        print("checkpoint saved.")

    def adjust_learning_rate(self, epoch):
        lr = 0.001
        if epoch > 60:
            lr = lr / 1000
        elif epoch > 40:
            lr = lr / 100
        elif epoch > 20:
            lr = lr / 10

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def train(self, num_epochs=60):
        best_acc = 0.0

        train_loader = DataLoader(dataset=self.trainData, batch_size=self.opt['batch_size'], shuffle=True, num_workers=2)

        for epoch in range(num_epochs):
            self.model.train()

            forg_acc = 0.0
            user_acc = 0.0
            train_loss = 0.0

            for _, (images, labels, forgeries) in enumerate(train_loader):
                if self.cuda_avail:
                    images, labels, forgeries = Variable(images.cuda(), requires_grad=True), Variable(labels.cuda(), requires_grad=True), Variable(forgeries.cuda(), requires_grad=True)
                else:
                    images, labels, forgeries = Variable(images, requires_grad=True), Variable(labels, requires_grad=True), Variable(forgeries, requires_grad=True)

                labels = labels.type(torch.FloatTensor)
                forgeries = forgeries.type(torch.FloatTensor)

                self.optimizer.zero_grad()
                label_pred, forg_pred = self.model(images)
                loss = self.loss(label_pred, forg_pred, labels, forgeries)

                loss.backward()
                self.optimizer.step()
                train_loss += loss.cpu().item() * images.size(0)

                _, prediction_user = torch.max(label_pred.data, 1)
                prediction_forg = []
                for row in forg_pred.data:
                    prediction_forg.append(1) if row.item() > 0.5 else prediction_forg.append(0)
                prediction_forg = torch.from_numpy(np.asarray(prediction_forg))

                # user acc
                prediction_user = prediction_user.type(torch.IntTensor)
                targets = torch.from_numpy(np.asarray([np.where(r==1)[0][0] for r in labels])).type(torch.IntTensor)
                user_acc += torch.sum(prediction_user == targets).cpu().item()

                # forg_acc
                forgery = forgeries.type(torch.IntTensor)
                forg_acc += torch.sum(prediction_forg == forgery).cpu().item()

            self.adjust_learning_rate(epoch)

            user_acc = user_acc / len(train_loader.dataset)
            forg_acc = forg_acc / len(train_loader.dataset)
            train_loss = train_loss / len(train_loader.dataset)

            test_acc, test_loss = self.test()

            print("Epoch {0:03d}, user_acc: {1:.4f}, forg_acc: {2:.4f} train_loss: {3:.4f}, test_acc: {4:.4f}, val_loss: {5:.4f}".format(epoch, user_acc, forg_acc, train_loss, test_acc, test_loss))

            if test_acc > best_acc:
                self.save_models(epoch)
                best_acc = test_acc

    def test(self):
        test_acc = 0.0
        test_loss = 0.0

        test_loader = DataLoader(dataset=self.testData, batch_size=self.opt['batch_size'], shuffle=True, num_workers=2)

        for i, (images, labels, forgeries) in enumerate(test_loader):
            if self.cuda_avail:
                images, labels, forgeries = Variable(images.cuda()), Variable(labels.cuda()), Variable(forgeries.cuda())
            else:
                images, labels, forgeries = Variable(images), Variable(labels), Variable(forgeries)

            label_pred, forg_pred = self.model(images)
            loss = self.loss(label_pred, forg_pred, labels, forgeries)
            test_loss += loss.cpu().item() * images.size(0)
            _, prediction = torch.max(label_pred.data, 1)

            prediction = prediction.type(torch.IntTensor)
            targets = torch.from_numpy(np.asarray([np.where(r==1)[0][0] for r in labels])).type(torch.IntTensor)
            test_acc += torch.sum(prediction == targets).cpu().item()

        test_acc = test_acc / len(test_loader.dataset)
        test_loss = test_loss / len(test_loader.dataset)

        return test_acc, test_loss
