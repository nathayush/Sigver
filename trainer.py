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


    # def loss(self, outputs, labels):
    #     # outputs[0] --- probability of image belonging to user classes --- dimension = num_images x num_classes --- P(y_j|X_i)
    #     # outputs[1] --- probability of image being a forgery --- dimension = num_images x 1 --- P(f|X_i)
    #     # labels[0] --- label of each user is a 1-d 2-tensors of target_user, forgery_bool --- dimension = num_users x 1 --- y_j, f_i
    #     out = 0
    #     # iterate over images
    #     for i in range(0, labels.shape[0]):
    #         temp = (labels[i][1] - 1).type(torch.FloatTensor)*torch.Tensor(np.array(1 - self.opt['lambda']))
    #         temp2 = 0
    #         # for each user for image
    #         for j in range(0, self.num_classes):
    #             # only if target of image is the jth user
    #             if j == labels.data.cpu().numpy()[i][0]: # labels[i][0] gives target user for image i
    #                 temp2 += log(outputs[0][i][j]) # outputs[0][i][j] means probability for image i and user j
    #         temp3 = labels[i][1].type(torch.FloatTensor)*log(outputs[1][i][0]) # outputs[1][i][0] means probability that image i is a forgery
    #         temp4 = (1 - labels[i][1]).type(torch.FloatTensor)*log(1-outputs[1][i][0]) # labels[i][1] gives forgery_bool for image i
    #         out += temp*temp2 - self.opt['lambda']*(temp3 + temp4)
    #         # out += (labels[i][1] - 1)*(1 - self.opt['lambda'])*temp2 - self.opt['lambda']*((labels[i][1]*np.log(outputs[1][i][0])) + ((1 - labels[i][1])*np.log(1-outputs[1][i][0])))
    #     return out

    def create_model(self):
        self.model = model.Signet(self.num_classes)

        if self.cuda_avail:
            self.model.cuda()

        self.temp_loss = torch.nn.CrossEntropyLoss()
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

            train_acc = 0.0
            train_loss = 0.0
            for _, (images, labels) in enumerate(train_loader):
                if self.cuda_avail:
                    images, labels = Variable(images.cuda(), requires_grad=True), Variable(labels.cuda(), requires_grad=True)
                else:
                    images, labels = Variable(images, requires_grad=False), Variable(labels, requires_grad=False)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                # loss = self.loss(outputs, labels)
                loss = self.temp_loss(outputs, labels.type(torch.LongTensor))
                loss.backward()
                self.optimizer.step()

                train_loss += loss.cpu().item() * images.size(0)
                _, prediction = torch.max(outputs.data, 1)

                prediction = prediction.type(torch.IntTensor)
                # targets = torch.Tensor(pd.DataFrame(labels.data.cpu().numpy())[0].values).type(torch.IntTensor)
                # train_acc += torch.sum(prediction == targets)
                train_acc += torch.sum(prediction.data == labels.data).cpu().item()

            self.adjust_learning_rate(epoch)

            train_acc = train_acc / len(train_loader.dataset)
            train_loss = train_loss / len(train_loader.dataset)

            test_acc, test_loss = self.test()

            print("Epoch {0:03d}, Train Accuracy: {1:.4f} , Train Loss: {2:.4f} , Test Accuracy: {3:.4f} , Val Loss: {4:.4f}".format(epoch, train_acc, train_loss, test_acc, test_loss))

            if test_acc > best_acc:
                self.save_models(epoch)
                best_acc = test_acc

    def test(self):
        test_acc = 0.0
        test_loss = 0.0

        test_loader = DataLoader(dataset=self.testData, batch_size=self.opt['batch_size'], shuffle=True, num_workers=2)

        for i, (images, labels) in enumerate(test_loader):
            if self.cuda_avail:
                images, labels = Variable(images.cuda()), Variable(labels.cuda())
            else:
                images, labels = Variable(images), Variable(labels)

            outputs = self.model(images)
            loss = self.temp_loss(outputs, labels.type(torch.LongTensor))
            test_loss += loss.cpu().item() * images.size(0)
            _, prediction = torch.max(outputs.data, 1)

            prediction = prediction.type(torch.IntTensor)
            test_acc += torch.sum(prediction.data == labels.data).cpu().item()

        test_acc = test_acc / len(test_loader.dataset)
        test_loss = test_loss / len(test_loader.dataset)

        return test_acc, test_loss
