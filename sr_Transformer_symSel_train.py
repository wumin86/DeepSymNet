# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
#import torch.optim.lr_scheduler.StepLR
#from __future__ import print_function
import torch.utils.data as data
from functools import cmp_to_key
from transformerModel import TransformerModel
from sr_Transformer_generateData import LoadData, LoadIsTestInTrain, outPutSymRlt
padToken = 256
HiddenNum = -1 #-1混合训练, 大于0固定当前隐层数训练
class MyDataset(data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __getitem__(self, index):#返回的是tensor
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

def GetEqual(symSel, varConstNum, symNetHiddenNum, connectSel):
    paramNumList = [2, 2, 2, 2, 1, 1, 1, 1]
    paramNums = torch.tensor(paramNumList, dtype=torch.long)
    symTable = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80]
    lenTable = len(symTable)
    symNum = lenTable

    indexCurr = torch.zeros(symNetHiddenNum, lenTable, 2, dtype=int)
    indexCurr[:] = -1
    count = 0
    lastSym = -1
    lastLayer = -1
    for i in range(symNetHiddenNum):
        if 0 == symSel[i] or padToken == symSel[i]:
            continue
        lastLayer = i
        for j in range(lenTable):
            if 0 < symSel[i] & symTable[j]:
                lastSym = j
                index1 = connectSel[count]
                count = count + 1
                if index1 < varConstNum + i * lenTable:
                    indexCurr[i, j, 0] = index1
                index2 = padToken #50
                if 2 == paramNums[j]:
                    index2 = connectSel[count]
                    count = count + 1
                if index2 < varConstNum + i * lenTable:
                    indexCurr[i, j, 1] = index2
                # else:
                #     indexCurr[i, j, 0] = -1
    finalIndex = lastSym + (symNetHiddenNum-lastLayer-1)*lenTable
    minSymFlag = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    symUseCountListTmp = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    equal = outPutSymRlt(variNum, symNum, symNetHiddenNum - 1, paramNums, finalIndex, indexCurr,  minSymFlag, symUseCountListTmp)
    equal = ''.join(equal[0])

    return  equal

def test(model, max_len, device, IsTestInTrain, constValueTest, testData, labelTest1, labelTest2):
    model.eval()
    with torch.no_grad():
        varConstNum = testData.size(2)
        test_times = testData.size(0)
        count = 0
        candNum = 1
        notInTrainSucNum = 0
        testInTrainNum = 0
        for i in range(test_times):
            isInTrain = IsTestInTrain[i, 0]
            if 0 < isInTrain:
                testInTrainNum = testInTrainNum + 1
            src = testData[i].unsqueeze(1).to(device)
            tgt = labelTest1[i].to(device)
            predList = []
            for k in range(candNum):
                pred = [0]
                for j in range(max_len):
                    inp = torch.LongTensor(pred).unsqueeze(1).to(device)
                    output = model(src, inp)
                    out_num = output.argmax(2)[-1].item()

                    if 0 == j:
                        for s in range(k):
                            output[0,0,out_num] = -100
                            out_num = output.argmax(2)[-1].item()
                    pred.append(out_num)
                predList.append(torch.tensor(pred[1:max_len + 1]).to(device))
            success = 0
            for k in range(len(predList)):
                if tgt.equal(predList[k]):
                    if 0 == isInTrain:
                        notInTrainSucNum = notInTrainSucNum + 1
                        print("haha!!!")
                    count = count+1
                    success = 1
                    break
            equal = GetEqual(tgt.cpu().numpy().tolist(), varConstNum, 6, labelTest2[i])
            constValueTmp = [float('{:.4f}'.format(i)) for i in constValueTest[i].numpy().tolist()]
            print(i, "isInTrain:", int(isInTrain), " success:", success, " target:", tgt.cpu().numpy().tolist(),
                  " predict:", predList[0].cpu().numpy().tolist(), "equal:", equal, "const:", constValueTmp)
        print("testInTrainNum", testInTrainNum, " notInTrainSucNum:", notInTrainSucNum, "Predict correct ratio:", count/test_times)

def ReGetCand(predList, probList, maxNum):
    num = len(probList)
    probTmp = torch.zeros(num, 2)
    for i in range(num):
        probTmp[i, 0] = i
        probTmp[i, 1] = probList[i]
    probTmp = probTmp[probTmp[:, 1].sort(descending=True)[1]]
    num = min(num, maxNum)
    predListTmp = []
    for i in range(num):
        index = int(probTmp[i, 0])
        predListTmp.append(predList[index])
    return predListTmp

def test_getSym(model, max_len1, maxCandNum, src, device):
    symTable = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80]
    predList = []
    pred = [0]
    predList.append(pred)
    probList = [1]
    for j in range(max_len1+1):#这里循环次数应该加1
        predListNew = []
        probListNew = []
        currTotalCandNum = len(predList)
        outValueAll = None
        finishList = []
        for k in range(currTotalCandNum):
            lenth = len(predList[k])
            if 1 < lenth and 0 == predList[k][lenth - 1]:
                finishList.append(k)
                continue
            inp = torch.LongTensor(predList[k]).unsqueeze(1).to(device)
            output = model(src, inp)
            outDim = output.size(2)
            outLen = output.size(0)
            # 不做归一化，因为做归一化效果反而不好
            norm_data = output[outLen - 1, 0, :] / 10.0

            if not (predList[k][lenth - 1] in symTable): #防止提前结束
                norm_data[0] = -100
            norm_data[256: outDim] = -100
            norm_data = norm_data.unsqueeze(1)
            indexTmp = torch.zeros(outDim, 2).to(device)
            for t in range(outDim):
                indexTmp[t, 0] = k
                indexTmp[t, 1] = t
            outValues = torch.cat([indexTmp, norm_data], 1)
            outValues = outValues[outValues[:, 2].sort(descending=True)[1]]

            numOk = 0  # 有效节点个数
            for i in range(outDim):
                if 0 >= outValues[i, 2]:
                    break
                numOk = numOk + 1
            if 0 == numOk:
                continue
            outValues = outValues[0:numOk, :]

            outValues[:, 2] = outValues[:, 2] * probList[k]
            num = min(maxCandNum, numOk)
            if outValueAll is None:
                outValueAll = outValues[0:num, :]
            else:
                outValueAll = torch.cat([outValueAll, outValues[0:num, :]], 0)
        if outValueAll is None:
            break
        outValueAll = outValueAll[outValueAll[:, 2].sort(descending=True)[1]]
        num = outValueAll.size(0)
        currFinishCandNum = len(finishList)
        num = min(num, maxCandNum - currFinishCandNum)
        for t in range(num):
            orgCandNo = int(outValueAll[t, 0])  # 原来所在的候选列表
            n = int(outValueAll[t, 1])  # 下一个元素
            newPred = predList[orgCandNo][:] #wumin 注意这里要用切片赋值
            newPred.append(n)
            predListNew.append(newPred)
            probListNew.append(outValueAll[t, 2])
        for t in range(currFinishCandNum):
            index = finishList[t]
            predListNew.append(predList[index])
            probListNew.append(probList[index])

        predList = predListNew
        probList = probListNew
    # 检查没有遇到分隔符的情况,下面的处理方式还比较简单，以后遇到没有分隔符的要对列表进行修改，使得必须包含间隔符，而不是简单丢弃
    predListNew = []
    probListNew = []
    for k in range(len(predList)):
        lenth = len(predList[k])
        if 0 != predList[k][lenth-1]:
            continue  #最后如果不是0的直接丢弃, 因为上面循环加1了
            if (predList[k][lenth-1] in symTable):
                predList[k].append(0)
                predListNew.append(predList[k])
                probListNew.append(probList[k])
        else:
            predListNew.append(predList[k])
            probListNew.append(probList[k])
    predList = predListNew
    probList = probListNew

    return predList, probList

def train_model(model, trainloader, valLoader, device):
    with torch.autograd.set_detect_anomaly(True):
        optimizer = optim.Adam(model.parameters(), lr=0.0001) #运算符选择学习率先0.0001再0.00001效果好
        #optimizer = optim.Adam(model.parameters(), lr=0.0002, weight_decay=0.001)#加上L2正则化
        #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        criterion = nn.CrossEntropyLoss()
        best_loss = 100
        for i in range(1):
            for epoch in range(500):
                epoch_loss = train(model, criterion, optimizer, trainloader, device)
                epoch_loss_val = validation(model, criterion, valLoader, device)
                print("i: {} epoch: {} train loss: {} val loss: {}".format(i, epoch, epoch_loss, epoch_loss_val))
                #print("epoch: {} val loss: {}".format(epoch, epoch_loss_val))
                if epoch_loss_val < best_loss:
                    best_loss = epoch_loss_val
                    model_name = "model_symSel/model_{0:.5f}.pt".format(epoch_loss_val)
                    torch.save(model.state_dict(), model_name)
        return model_name

def GetLabelLen(labelData):
    histLen = [0 for i in range(7)]
    num = labelData.size(0)
    for i in range(num):
        tmp = labelData[i,:].nonzero()
        n = tmp.size(0)
        histLen[n] = histLen[n] + 1
    return histLen

#3 < HiddenNum才调用
def ReGetLabel1_old(label1):
    label1[:, HiddenNum + 1:symNetHiddenNum] = padToken
    if symNetHiddenNum == HiddenNum:  # 还要考虑加0!
        tmpLabel1 = torch.zeros(label1.size(0), 1, dtype=int)
        label1 = torch.cat([label1, tmpLabel1], 1)
    return  label1

#混合训练用
def ReGetLabel1(label1):
    num = label1.size(0)
    lenth = label1.size(1)
    tmpLabel1 = torch.zeros(num, 1, dtype=int)
    label1 = torch.cat([label1, tmpLabel1], 1)
    for i in range(num):
        lenthCurr = 0
        for j in range(lenth):
            if 0 == label1[i, j]:
                break
            lenthCurr = lenthCurr + 1
        label1[i, lenthCurr] = 0
        label1[i, lenthCurr + 1:lenth+1] = padToken
    return label1

def LoadModel(model, modelName):
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(modelName))
    else:
        model.load_state_dict(torch.load(modelName, map_location='cpu'))

if __name__ == '__main__':
    with torch.no_grad():
        #torch.manual_seed(123)  # 初始化种子后每次运行的结果都一样
        trainSampleNum = 500
        paramNumList = [2, 2, 2, 2, 1, 1, 1, 1]  # , 1]
        symNum = len(paramNumList)
        paramNums = torch.tensor(paramNumList, dtype=torch.long)
        variNum = 3
        constNum = 1
        variConstNum = variNum + constNum
        symNetHiddenNum = 6
        inputSeqLen = 20 #输入序列长度
        batch_size = 128
        #不需要调用CreatAllDataNew2生成数据，直接加载数据就行
        inputTrain, labelTrain1, labelTrain2, constValueTrain = LoadData("train")
        maxValue = inputTrain[:,:,2].max() #检查数据是否有非法的
        trainNum = labelTrain1.size(0)
        print("train sample num:", trainNum)
        histLenTrain = GetLabelLen(labelTrain1)
        print("histLenTrain:", histLenTrain)
        if 3 < HiddenNum:
            labelTrain1 = ReGetLabel1_old(labelTrain1)
        else:
            labelTrain1 = ReGetLabel1(labelTrain1)
            if 0 < HiddenNum:
                labelTrain1 = labelTrain1[:, 0:-1]
        trainloader = torch.utils.data.DataLoader(MyDataset(inputTrain, labelTrain1), batch_size*5, shuffle=True, num_workers=2)

        inputValid, labelValid1, labelValid2, constValueValid = LoadData("valid")
        validNum = labelValid1.size(0)
        print("valid sample num:", validNum)
        histLenValid = GetLabelLen(labelValid1)
        print("histLenValid:", histLenValid)
        if 3 < HiddenNum:
            labelValid1 = ReGetLabel1_old(labelValid1)
        else:
            labelValid1 = ReGetLabel1(labelValid1)
            if 0 < HiddenNum:
                labelValid1 = labelValid1[:, 0:-1]
        valLoader = torch.utils.data.DataLoader(MyDataset(inputValid, labelValid1), batch_size*1, shuffle=True, num_workers=2)

        inputTest, labelTest1, labelTest2, constValueTest = LoadData("test6")
        #IsTestInTrain = LoadIsTestInTrain()
        IsTestInTrain = LoadIsTestInTrain("IsTestInTrain6")
        totalNumInTrain = IsTestInTrain.sum() # 测试数据在训练集中的个数

        testNum = labelTest1.size(0)
        print("test sample num:", testNum)
        histLenTest = GetLabelLen(labelTest1)
        print("histLenTest:", histLenTest)
        if 3 < HiddenNum:
            labelTest1[:, HiddenNum + 1:symNetHiddenNum] = padToken
        else:
            labelTest1 = ReGetLabel1(labelTest1)
            # test数据不用考虑后面加0!
            labelTest1 = labelTest1[:, 0:-1]

        print("Max value in inputTrain:", float(maxValue))
        print("totalNumInTrain:", int(totalNumInTrain))

        maxLen = symNetHiddenNum
        ntoken = 257 #256
        d_model = 512#默认512   256
        nlayers = 6 #默认6       2
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerModel(variNum+1, ntoken, d_model, nlayers=nlayers).to(device)
    #LoadModel(model, "model/model_1.16175.pt")
    model_name = train_model(model, trainloader, valLoader, device)

    print('Finish Training')