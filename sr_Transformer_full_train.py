# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
#import torch.optim.lr_scheduler.StepLR
#from __future__ import print_function
import torch.utils.data as data
from functools import cmp_to_key
import copy
from transformerModel import TransformerModel
from sr_Transformer_generateData import LoadData, LoadIsTestInTrain, outPutSymRlt, GetCurrIndex, GetInputSeq
from scipy.optimize import least_squares, minimize
import sympy


import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

padToken = 256
HiddenNum = -1 #-1混合训练, 大于0固定当前隐层数训练
class MyDataset(data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __getitem__(self, index):#返回的是tensor
        # a = torch.tensor(0).unsqueeze(0)
        # b = self.labels[index]
        # c = self.images[index]
        d = torch.cat([torch.tensor(0).unsqueeze(0),self.labels[index]], 0)
        img, target = self.images[index], d
        return img, target

    def __len__(self):
        return len(self.images)

def train(model, criterion, optimizer, trainloader, device):
    with torch.autograd.set_detect_anomaly(True):
        model.train()  # turn on train mode
        epoch_loss = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.transpose(1, 0), labels.transpose(1, 0)
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs, labels[:-1, :])
            n = outputs.shape[-1]
            loss = criterion(outputs.reshape(-1, n), labels[1:, :].reshape(-1))
            #loss = criterion(outputs.view(-1, ntokens), labels)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

            optimizer.step()
            #epoch_loss += loss.item()
        epoch_loss = epoch_loss / len(trainloader)
        return epoch_loss

def validation(model, criterion, valLoader, device):
    with torch.autograd.set_detect_anomaly(True):
        model.eval()
        epoch_loss = 0
        with torch.no_grad():
            for i, data in enumerate(valLoader, 0):
                # get the inputs
                inputs, labels = data
                inputs, labels = inputs.transpose(1, 0), labels.transpose(1, 0)
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs, labels[:-1, :])
                n = outputs.shape[-1]
                loss = criterion(outputs.reshape(-1, n), labels[1:, :].reshape(-1))
                #loss = criterion(outputs.view(-1, ntokens), labels)
                epoch_loss += loss.item()
            epoch_loss = epoch_loss / len(valLoader)
            return epoch_loss

#####################################################################################
def train_model(model, trainloader, valLoader, device):
    with torch.autograd.set_detect_anomaly(True):
        optimizer = optim.Adam(model.parameters(), lr=0.00001)#全模型学习率一开始就0.00001效果好
        #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        criterion = nn.CrossEntropyLoss(ignore_index = padToken)
        best_loss = 100
        for i in range(1):
            for epoch in range(500):
                epoch_loss = train(model, criterion, optimizer, trainloader, device)
                epoch_loss_val = validation(model, criterion, valLoader, device)
                print("i: {} epoch: {} train loss: {} val loss: {}".format(i, epoch, epoch_loss, epoch_loss_val))
                #print("epoch: {} val loss: {}".format(epoch, epoch_loss_val))
                if epoch_loss_val < best_loss:
                    best_loss = epoch_loss_val
                    model_name = "model/model_{0:.5f}.pt".format(epoch_loss_val)
                    torch.save(model.state_dict(), model_name)
        return model_name

def GetLabelLen(labelData, symNetHiddenNum):
    histLen = [0 for i in range(symNetHiddenNum+1)]#wumin这里的7改成参数symNetHiddenNum+1
    num = labelData.size(0)
    for i in range(num):
        tmp = labelData[i,:].nonzero()
        n = tmp.size(0)
        histLen[n] = histLen[n] + 1
    return histLen

def MergeLabel(maxLabelLen, label1, label2):
    num = label1.size(0)
    dim1 = label1.size(1)
    dim2 = label2.size(1)
    label = torch.ones(num, maxLabelLen, dtype=int)*padToken
    for i in range(num):
        for j in range(dim1):
            if 0 == label1[i, j]:
                label[i, 0:j+1] = label1[i, 0:j+1]
                break
        len1 = j+1
        if 0 != label1[i, dim1-1]: #这种情况说明label1没有0, 从而label也没有赋值
            label[i, 0:dim1] = label1[i, 0:dim1]
            label[i, dim1] = 0
            len1 = dim1 + 1
        label[i, len1:len1 + dim2] = label2[i]
    return label

def LoadModel(model, modelName):
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(modelName))
    else:
        model.load_state_dict(torch.load(modelName, map_location='cpu'))

if __name__ == '__main__':
    with torch.no_grad():

        variNum = 3

        symNetHiddenNum = 6
        maxLabelLen2 = 24 #12, 16, 20
        if 0 < HiddenNum:
            maxLabelLen2 = 12 + (HiddenNum-3)*4
        maxLabelLen = symNetHiddenNum + maxLabelLen2 + 2 #加2是因为有一个间隔符0和终止符 padToken
        inputSeqLen = 20 #输入序列长度
        batch_size = 12

        inputTrain, labelTrain1, labelTrain2, constValueTrain = LoadData("train")
        labelTrain = MergeLabel(maxLabelLen, labelTrain1, labelTrain2)

        maxValue = inputTrain[:,:,2].max() #检查数据是否有非法的

        trainloader = torch.utils.data.DataLoader(MyDataset(inputTrain, labelTrain), batch_size*1, shuffle=True, num_workers=2)
        trainNum = labelTrain1.size(0)
        print("train sample num:", trainNum)
        histLenTrain = GetLabelLen(labelTrain1, symNetHiddenNum)
        print("histLenTrain:", histLenTrain)

        inputValid, labelValid1, labelValid2, constValueValid = LoadData("valid")
        labelValid = MergeLabel(maxLabelLen, labelValid1, labelValid2)
        valLoader = torch.utils.data.DataLoader(MyDataset(inputValid, labelValid), batch_size, shuffle=True, num_workers=2)
        validNum = labelValid1.size(0)
        print("valid sample num:", validNum)
        histLenValid = GetLabelLen(labelValid1, symNetHiddenNum)
        print("histLenValid:", histLenValid)


        ntoken = 257 #256
        d_model = 512#默认512   256
        nlayers = 6 #默认6       2
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerModel(variNum+1, ntoken, d_model, nlayers=nlayers).to(device)
    #LoadModel(model, "model/model_0.xxxxx.pt")
    model_name = train_model(model, trainloader, valLoader, device)

    print('Finish Training!')