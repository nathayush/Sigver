import model
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import pandas as pd
import os
import shutil
import time

class Trainer:
    def __init__(self, trainData, testData):
        self.trainData = trainData
        self.testData = testData
        self.cuda_avail = torch.cuda.is_available()

        self.opt = {
            'batch_size': 32
        }

        self.create_model()

    def create_model(self):
        self.model = model.Signet()

        if self.cuda_avail:
            self.model.cuda()

        self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=1e-3, weight_decay=1e-4)
        self.optimizer1 = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=1e-3, weight_decay=1e-4)
        self.loss_fn = torch.nn.BCELoss()

    def clear_model_checkpoints(self):
            if os.path.isdir("checkpoints"):
                shutil.rmtree("checkpoints")
                os.makedirs("checkpoints")
            else:
                os.makedirs("checkpoints")

    def save_model(self, epoch):
        torch.save(self.model.state_dict(), "checkpoints/Signet_{}.pkl".format(epoch))
        print("checkpoint saved.")

    def adjust_learning_rate(self, epoch):
        lr = 1e-3
        if epoch > 60:
            lr = lr / 1000
        elif epoch > 40:
            lr = lr / 100
        elif epoch > 20:
            lr = lr / 10

        for param_group in self.optimizer1.param_groups:
            param_group["lr"] = lr

    def train(self, num_epochs=60):
        best_acc = 0.0

        train_loader = DataLoader(dataset=self.trainData, batch_size=self.opt['batch_size'], shuffle=True, num_workers=2)

        for epoch in range(num_epochs):
            start_time = time.time()

            self.model.train()

            forg_acc = 0.0
            train_loss = 0.0

            for _, (images, forgeries) in enumerate(train_loader):
                if self.cuda_avail:
                    images, forgeries = Variable(images.cuda()), Variable(forgeries.cuda())
                else:
                    images, forgeries = Variable(images), Variable(forgeries)

                self.optimizer1.zero_grad()
                forg_pred = self.model(images)
                forgeries = forgeries.unsqueeze_(1)

                loss = self.loss_fn(forg_pred, forgeries)
                loss.backward()
                self.optimizer1.step()
                train_loss += loss.cpu().item() * images.size(0)

                forg_pred = torch.ge(forg_pred.data, 0.7).type(torch.FloatTensor)
                forg_acc += torch.sum(forg_pred == forgeries).cpu().item()

            self.adjust_learning_rate(epoch)

            forg_acc = forg_acc / len(train_loader.dataset)
            train_loss = train_loss / len(train_loader.dataset)

            print("Epoch {0:02d}, train_acc: {1:.4f}, train_loss: {2:.4f}, time: {3:.2f} sec".format(epoch, forg_acc, train_loss, time.time()-start_time))

            test_acc = self.eval()

            if test_acc > best_acc:
                self.save_model(epoch)
                best_acc = test_acc

    def eval(self):
        start_time = time.time()

        self.model.eval()
        test_acc = 0.0
        test_loss = 0.0

        test_loader = DataLoader(dataset=self.testData, batch_size=self.opt['batch_size'], shuffle=True, num_workers=2)
        for i, (images, forgeries) in enumerate(test_loader):
            if self.cuda_avail:
                images, forgeries = Variable(images.cuda()), Variable(forgeries.cuda())
            else:
                images, forgeries = Variable(images), Variable(forgeries)

            forg_pred = self.model(images)
            forgeries = forgeries.unsqueeze_(1)

            loss = self.loss_fn(forg_pred, forgeries)
            test_loss += loss.cpu().item() * images.size(0)

            forg_pred = torch.ge(forg_pred.data, 0.6).type(torch.FloatTensor)
            test_acc += torch.sum(forg_pred == forgeries).cpu().item()

        test_acc = test_acc / len(test_loader.dataset)
        test_loss = test_loss / len(test_loader.dataset)

        print("val_acc: {0:.4f}, val_loss: {1:.4f}, time: {2:.2f} sec".format(test_acc, test_loss, time.time()-start_time))

        return test_acc

    def test(self, data):
        start_time = time.time()

        self.model.eval()
        test_acc = 0.0
        tp = 0.0
        tn = 0.0
        fp = 0.0
        fn = 0.0

        test_loader = DataLoader(dataset=data, batch_size=self.opt['batch_size'], shuffle=True, num_workers=2)
        for i, (images, forgeries) in enumerate(test_loader):
            if self.cuda_avail:
                images, forgeries = Variable(images.cuda()), Variable(forgeries.cuda())
            else:
                images, forgeries = Variable(images), Variable(forgeries)

            forg_pred = self.model(images)
            forgeries = forgeries.unsqueeze_(1)
            forg_pred = torch.ge(forg_pred.data, 0.6).type(torch.FloatTensor)

            test_acc += torch.sum(forg_pred == forgeries).cpu().item()
            tp += len(torch.nonzero(forg_pred * forgeries))
            tn += len(torch.nonzero((forg_pred - 1) * (forgeries - 1)))
            fp += len(torch.nonzero(forg_pred * (forgeries - 1)))
            fn += len(torch.nonzero((forg_pred - 1) * forgeries))

        test_acc = test_acc / len(test_loader.dataset)

        print("test_acc: {0:.4f}, time: {1:.2f} sec".format(test_acc, time.time()-start_time))

        print("precision: {0:.2f}".format(tp / (tp + fp)))
        print("recall: {0:.2f}".format(tp / (tp + fn)))
        print("f1 score: {0:.2f}".format(2 * (tp / (tp + fp)) * (tp / (tp + fn)) / ((tp / (tp + fp)) + (tp / (tp + fn)))))
