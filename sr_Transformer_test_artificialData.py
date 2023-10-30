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
from sr_Transformer_generateData import LoadData, LoadIsTestInTrain, outPutSymRlt, func
from scipy.optimize import minimize
import sympy


import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

padToken = 256
HiddenNum = -1 #-1 mixed training, greater than 0 indicates fixed current hidden layer training

def GetEqual(symSel, variConstNum, symNetHiddenNum, connectSel):
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
        if 0 == symSel[i]:
            continue
        lastLayer = i
        for j in range(lenTable):
            if 0 < symSel[i] & symTable[j]:
                lastSym = j
                index1 = connectSel[count]
                count = count + 1
                if index1 < variConstNum + i * lenTable:
                    indexCurr[i, j, 0] = index1
                index2 = padToken # 50
                if 2 == paramNums[j]:
                    index2 = connectSel[count]
                    count = count + 1
                if index2 < variConstNum + i * lenTable:
                    indexCurr[i, j, 1] = index2
                # else:
                #     indexCurr[i, j, 0] = -1
    finalIndex = lastSym + (symNetHiddenNum-lastLayer-1)*lenTable
    minSymFlag = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    symUseCountListTmp = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    equal = outPutSymRlt(variNum, symNum, symNetHiddenNum - 1, paramNums, finalIndex, indexCurr,  minSymFlag, symUseCountListTmp)
    equal = ''.join(equal[0])

    return  equal

#######################################
def GetNextLayerDataConstTmp(symNum, inputData, allIndex, constNo, constValue, validFlagIn):
    a = inputData.size()[0]# 数据维度
    x = torch.zeros(a)# x的赋值可能要通过data
    preNodeNum = validFlagIn.size()[0]
    validFlagOut = torch.zeros(symNum+preNodeNum)

    for i in range(0, symNum):
        idx1 = allIndex[i, 0]
        idx2 = allIndex[i, 1]
        if idx1 == -1:
            continue
        if validFlagIn[idx1] == 0 or (idx2 != -1 and validFlagIn[idx2] == 0):
            allIndex[i, 0] = -1
            allIndex[i, 1] = -1
            continue

        validFlagOut[i] = 1

        if preNodeNum-1 == idx1: #0 == (idx1-variNum)%symNum:
            inputData[idx1] = constValue[constNo]
            constNo = constNo + 1
        if -1 != idx2 and preNodeNum-1 == idx2: #0 == (idx2-variNum)%symNum:
            inputData[idx2] = constValue[constNo]
            constNo = constNo + 1

        if i == 0:
            x[i] = (inputData[idx1] + inputData[idx2])

        if i == 1:
            x[i] = (inputData[idx1] - inputData[idx2])

        if i == 2:
            x[i] = (inputData[idx1] * inputData[idx2])
            if abs(x[i]) > 1000:
                allIndex[i, 0] = -1
                allIndex[i, 1] = -1
                validFlagOut[i] = 0
                #print(x[i], " multi caution!!!")

        if i == 3:
            if inputData[idx2] != 0:
                x[i] = (inputData[idx1] / inputData[idx2])
                if abs(x[i]) > 1000:
                    allIndex[i, 0] = -1
                    allIndex[i, 1] = -1
                    validFlagOut[i] = 0
                    #print(x[i], " div caution!!!")
            else:
                allIndex[i, 0] = -1
                allIndex[i, 1] = -1
                validFlagOut[i] = 0

        if i == 4:
            x[i] = torch.sin(inputData[idx1])

        if i == 5:
            x[i] = torch.cos(inputData[idx1])

        if i == 6:
            if inputData[idx1] < 5:
                x[i] = torch.exp(inputData[idx1])
            else:
                allIndex[i, 0] = -1
                allIndex[i, 1] = -1
                validFlagOut[i] = 0
                #print(x[i], " exp caution!!!")

        if i == 7:
            if inputData[idx1] > 0:
                x[i] = torch.log(inputData[idx1])
                if abs(x[i])>1000:
                    allIndex[i, 0] = -1
                    allIndex[i, 1] = -1
                    validFlagOut[i] = 0
                    #print(x[i]," log caution!!!")
            else:
                allIndex[i, 0] = -1
                allIndex[i, 1] = -1
                validFlagOut[i] = 0
        if i == 8:
            x[i] = inputData[idx1]*inputData[idx1] #torch.pow(inputData[idx1], 2)

    for i in range(0, preNodeNum):
        x[i+symNum] = inputData[i]
        validFlagOut[i+symNum] = validFlagIn[i]

    return x, validFlagOut, constNo

def error(p, x, y, codeStr):
    """
    Fitting residuals.
    """
    return y - func(p, x, codeStr)

def objFunc(p, x, y, codeStr):
    """
    Fitting residuals.
    """
    #x, y, codeStr = args
    z = y - func(p, x, codeStr)
    totalError = sum(z ** 2)
    return totalError

def fit_func(data_x, data_y, codeStr, TestData, pred):
    """
    这里 p0 存放的是c1、c2的初始值，这个值会随着拟合的进行不断变化，使得误差
    error的值越来越小
    """
    minError = 9999
    minConstList = []
    for i in range(10):
        if 0 == i:
            p0 = np.random.rand(10) # np.zeros(10)
        else:
            p0 = np.random.rand(10)

        fit_res = minimize(objFunc, p0, args=(data_x, data_y, codeStr), method='BFGS', options={'gtol': 1e-6, 'disp': False})

        rlt = dict(fit_res)

        totalError = sum(abs(error(rlt["x"], data_x, data_y, codeStr))) / 20
        if totalError < minError:
            minError = totalError
            minConstList = rlt["x"]
        if minError < 1.0e-8:
            break

    #print("    求得的系数为:", minConstList, ", totalError:", minError)
    return minConstList, minError

def GetConstValue(variConstNum, symNetHiddenNum, TestData, pred):
    symSelPred = [0 for i in range(symNetHiddenNum)]
    for i in range(len(pred)):
        if 0 == pred[i + 1]:
            break
        symSelPred[i] = pred[i + 1]
    predConnect = pred[i + 1 + 1:]
    codeStr = GetEqual(symSelPred, variConstNum, symNetHiddenNum, predConnect)
    # data_x = TestData[:, 0:2].numpy()
    # data_y = TestData[:, 2].numpy()
    # constList, totalError = fit_func(data_x, data_y, codeStr, TestData, pred)
    a = codeStr.count('const')
    constList = []
    totalError = 9999
    if a <= 10:
        data_x = TestData[:, 0:variConstNum - 1].numpy()
        data_y = TestData[:, variConstNum - 1].numpy()
        constList, totalError = fit_func(data_x, data_y, codeStr, TestData, pred)


    return constList, totalError
#return only one result
def GetFinalCand(variConstNum, symNetHiddenNum, TestData, constValueTest, allPredList):
    candNum = len(allPredList)
    minTotalError = 9999
    minConstList = []
    minIndex = -1
    finalCandList = []
    for i in range(candNum):
        if padToken != allPredList[i][-1]:
            continue

        constList, totalError = GetConstValue(variConstNum, symNetHiddenNum, TestData, allPredList[i])
        if totalError < minTotalError:
            minTotalError = totalError
            minConstList = constList
            minIndex = i
    if -1 != minIndex:
        finalCandList = [allPredList[minIndex]]
    return finalCandList, minConstList, minTotalError

#####################################################################################
def PrintTgtPreEqual(variConstNum, symNetHiddenNum, tgt, pred):
    symSelTgt = [0 for i in range(symNetHiddenNum)]
    for i in range(len(tgt)):
        if 0 == tgt[i]:
            break
        symSelTgt[i] = int(tgt[i])
    tgtConnect = tgt[i+1:]

    symSelPred = [0 for i in range(symNetHiddenNum)]
    for i in range(len(pred)):
        if 0 == pred[i+1]:
            break
        symSelPred[i] = pred[i+1]
    predConnect = pred[i+1+1:]

    tgtEqual = GetEqual(symSelTgt, variConstNum, symNetHiddenNum, tgtConnect)
    preEqual = GetEqual(symSelPred, variConstNum, symNetHiddenNum, predConnect)
    print("    tgtEqual:", tgtEqual, "preEqual:", preEqual)

def test_getSym(model, max_len1, maxCandNum, src, device):
    symTable = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80]
    predList = []
    pred = [0]
    predList.append(pred)
    probList = [1]
    for j in range(max_len1+1):
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

            norm_data = output[outLen - 1, 0, :] / 10.0

            if not (predList[k][lenth - 1] in symTable): #Prevent early termination
                norm_data[0] = -100
            norm_data[256: outDim] = -100
            norm_data = norm_data.unsqueeze(1)
            indexTmp = torch.zeros(outDim, 2).to(device)
            for t in range(outDim):
                indexTmp[t, 0] = k
                indexTmp[t, 1] = t
            outValues = torch.cat([indexTmp, norm_data], 1)
            outValues = outValues[outValues[:, 2].sort(descending=True)[1]]

            numOk = 0  # Number of valid nodes
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
            orgCandNo = int(outValueAll[t, 0])  # Original candidate list
            n = int(outValueAll[t, 1])  # Next element
            newPred = predList[orgCandNo][:]
            newPred.append(n)
            predListNew.append(newPred)
            probListNew.append(outValueAll[t, 2])
        for t in range(currFinishCandNum):
            index = finishList[t]
            predListNew.append(predList[index])
            probListNew.append(probList[index])

        predList = predListNew
        probList = probListNew
    # If no delimiter is encountered, discard
    predListNew = []
    probListNew = []
    for k in range(len(predList)):
        lenth = len(predList[k])
        if 0 != predList[k][lenth-1]:
            continue
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

def get_perLayerSymList(predList):
    symTable = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80]
    lenTable = len(symTable)
    perLayerSymList = []
    for currCandPredList in (predList):
        perLayerSymListTmp = []
        for symNo in (currCandPredList):
            if 0 == symNo:
                continue
            symList = []
            for k in range(lenTable):
                if 0 < (symNo & symTable[k]):
                    symList.append(k)
            perLayerSymListTmp.append(symList)
        perLayerSymList.append(perLayerSymListTmp)
    return perLayerSymList

#新的
def GetOneConnectNode(model, maxCandNum, candNo, prob, pred, src, symNo, opNo, preNode, sinCosLogFlag, validFlag, device):
    preNodeNum = validFlag.size()[0] #Number of nodes in the previous layer
    count = 0 #Number of valid nodes in the previous layer
    for i in range(preNodeNum):
        if validFlag[i] == 1:
            count = count + 1
    if count == 0.0 or ((symNo == 1 or symNo == 3) and count == 1): #If there is a minus or division sign and only one valid node, do not select
        return -1, None

    inp = torch.LongTensor(pred).unsqueeze(1).to(device)
    output = model(src, inp)
    outDim = output.size(2)
    outLen = output.size(0)

    output[outLen-1, 0, preNodeNum:outDim] = -100
    for i in range(preNodeNum):
        if 0 == validFlag[i]:
            output[outLen-1, 0, i] = -100

    numFlag = 0 #Record the number of available sincoslog nodes
    for i in range(preNodeNum):
        if validFlag[i] == 0:
            continue
        if sinCosLogFlag[i] == 0:
            numFlag = numFlag + 1

    if symNo == 4 and symNo == 5 and symNo == 6 and symNo == 7 and 0 == numFlag:
        return -1, None

    for i in range(preNodeNum):
        if validFlag[i] == 0:
            continue
        # If it is a minus or division sign and the nodes are selected twice, do not select
        if (symNo == 1 or symNo == 3) and 1 == opNo and preNode == i:
            output[outLen - 1, 0, i] = -100
            continue
        if symNo != 4 and symNo != 5 and symNo != 6 and symNo != 7:
            continue
        if sinCosLogFlag[i] == 0:
            continue
        output[outLen-1, 0, i] = -100

    norm_data = output[outLen - 1, 0, 0:preNodeNum] #Only this part is valid
    norm_data = norm_data.unsqueeze(1)
    indexTmp = torch.zeros(preNodeNum, 2).to(device)
    for t in range(preNodeNum):
        indexTmp[t, 0] = candNo
        indexTmp[t, 1] = t
    outValues = torch.cat([indexTmp, norm_data], 1)
    outValues = outValues[outValues[:, 2].sort(descending=True)[1]]
    num = 0 #Number of valid nodes
    for i in range(preNodeNum):
        if 0 >= outValues[i, 2]:
            break
        num = num + 1
    if 0 == num:
        return -1, None
    outValues = outValues[0:num, :]
    outValues[:, 2] = (outValues[:, 2]/10.0) * prob
    #outValues = outValues[outValues[:, 2].sort(descending=True)[1]]
    num = min(num, maxCandNum)
    outValues = outValues[0:num, :]
    return 1, outValues

def UpdateLayerSymOpPreNode(layerSymOpList, perLayerSymList, orgCandNo, paramNums, preNode):
    layerNo = layerSymOpList[orgCandNo][0]
    symIndex = layerSymOpList[orgCandNo][1]
    symNo = perLayerSymList[orgCandNo][layerNo][symIndex]
    opNo = layerSymOpList[orgCandNo][2]
    paramNum = paramNums[symNo]

    currSymNum = len(perLayerSymList[orgCandNo][layerNo])
    layerNum = len(perLayerSymList[orgCandNo])

    layerSymOp = copy.deepcopy(layerSymOpList[orgCandNo])
    flag = 0 #0 hasn't changed layers, 1 has changed layers, 2 is over
    if currSymNum-1 == symIndex and paramNum-1 == opNo: #Change layer
        if layerNum-1 == layerNo: #Sequence generation completed
            flag = 2
        else: #Change layer
            flag = 1
            layerSymOp[0] = layerSymOp[0] + 1
            layerSymOp[1] = 0
            layerSymOp[2] = 0
            layerSymOp[3] = preNode
    else: #Do not Change layer
        if paramNum-1 == opNo:#New operator
            layerSymOp[1] = layerSymOp[1] + 1
            layerSymOp[2] = 0
        else:
            layerSymOp[2] = layerSymOp[2] + 1
        layerSymOp[3] = preNode

    return flag, layerSymOp

#Changed layers, updated sinCosLogFlagListNew, validFlagListNew
def UpdatesinCosLogValidFlag(layerSymOpList, perLayerSymList, predList, orgCandNo, paramNums, sinCosLogFlagList, validFlagList):
    layerNo = layerSymOpList[orgCandNo][0]
    currSymNum = len(perLayerSymList[orgCandNo][layerNo])

    symNum = len(paramNums)

    sinCosLogFlag = sinCosLogFlagList[orgCandNo]
    validFlag = validFlagList[orgCandNo]
    preNodeNum = sinCosLogFlag.size(0)
    currNodeNum = preNodeNum + symNum
    sinCosLogFlagOut = torch.zeros(currNodeNum)
    validFlagOut = torch.zeros(currNodeNum)
    currPred = predList[orgCandNo]
    currPredLen = len(currPred)
    lenth = 0
    for i in range(currSymNum):
        symNo = perLayerSymList[orgCandNo][layerNo][i]
        lenth = lenth + paramNums[symNo]
    strartIndex = currPredLen-lenth
    offSet = 0
    for i in range(currSymNum):
        symNo = perLayerSymList[orgCandNo][layerNo][i]
        validFlagOut[symNo] = 1
        index1 = currPred[strartIndex+offSet]
        offSet = offSet + 1
        index2 = -1
        if 2 == paramNums[symNo]:
            index2 = currPred[strartIndex+offSet]
            offSet = offSet + 1
        if 0 < sinCosLogFlag[index1] or (-1 != index2 and 0 < sinCosLogFlag[index2]):
            sinCosLogFlagOut[symNo] = 1
    for i in range(0, preNodeNum):
        sinCosLogFlagOut[i + symNum] = sinCosLogFlag[i]
        validFlagOut[i+symNum] = validFlag[i]

    return sinCosLogFlagOut, validFlagOut

def test_getConnect(model, variConstNum, paramNums, max_len2, maxCandNum, src, predList, probList, device):
    currTotalCandNum = len(predList)
    perLayerSymList = get_perLayerSymList(predList)
    sinCosLogFlagList = [torch.zeros(variConstNum) for i in range(currTotalCandNum)] #Need to update when changing layers
    validFlagList = [torch.ones(variConstNum) for i in range(currTotalCandNum)] #Need to update when changing layers
    layerSymOpList = [] #Every update required
    for i in range(currTotalCandNum):
        layerSymOp = torch.zeros(4, dtype=int)
        layerSymOp[0] = 0 #Layer No.
        layerSymOp[1] = 0 #operator index
        layerSymOp[2] = 0 #Which operand
        layerSymOp[3] = -1 #Store the previous node number preNode
        layerSymOpList.append(layerSymOp)

    for i in range(max_len2):
        predListNew = []
        probListNew = []
        finishList = []
        perLayerSymListNew = []
        sinCosLogFlagListNew = []
        validFlagListNew = []
        layerSymOpListNew = []
        currTotalCandNum = len(predList)
        outValueAll = None
        for j in range(currTotalCandNum):
            lenth = len(predList[j])
            if padToken == predList[j][lenth - 1]: #50
                finishList.append(j)
                continue
            layerNo = layerSymOpList[j][0]
            symIndex = layerSymOpList[j][1]
            symNo = perLayerSymList[j][layerNo][symIndex]
            opNo = layerSymOpList[j][2]
            preNode = layerSymOpList[j][3]
            prob = probList[j]
            flag, outValues = GetOneConnectNode(model, maxCandNum, j, prob, predList[j], src, symNo, opNo, preNode, sinCosLogFlagList[j],
                              validFlagList[j], device)
            if outValues is None:
                continue
            if outValueAll is None:
                outValueAll = outValues
            else:
                outValueAll = torch.cat([outValueAll, outValues], 0)
        if outValueAll is None:
            break
        outValueAll = outValueAll[outValueAll[:, 2].sort(descending=True)[1]]
        num = 0
        if not(outValueAll is None):
            num = outValueAll.size(0)
        currFinishCandNum = len(finishList)
        num = min(num, maxCandNum - currFinishCandNum)
        for t in range(num):
            orgCandNo = int(outValueAll[t, 0])  # Original candidate list
            n = int(outValueAll[t, 1])  # Next element
            newPred = predList[orgCandNo][:]
            newPred.append(n)
            probListNew.append(outValueAll[t, 2])
            perLayerSymListNew.append(perLayerSymList[orgCandNo])
            #LayerSymOpListNew needs to be updated every time, and when changing layers, sinCosLogFlagListNew and validFlagListNew are updated based on the selected situation
            flag, layerSymOp = UpdateLayerSymOpPreNode(layerSymOpList, perLayerSymList, orgCandNo, paramNums, n)
            layerSymOpListNew.append(layerSymOp)
            if 2 == flag:
                newPred.append(padToken) #50
            predListNew.append(newPred)

            if 1 == flag: #此时换层了
                sinCosLogFlag, validFlag = UpdatesinCosLogValidFlag(layerSymOpList, perLayerSymList, predList, orgCandNo, paramNums, sinCosLogFlagList, validFlagList)
            else:
                sinCosLogFlag = sinCosLogFlagList[orgCandNo]
                validFlag = validFlagList[orgCandNo]
            sinCosLogFlagListNew.append(sinCosLogFlag)
            validFlagListNew.append(validFlag)

        for t in range(currFinishCandNum):
            index = finishList[t]
            predListNew.append(predList[index])
            probListNew.append(probList[index])
            perLayerSymListNew.append(perLayerSymList[index])
            sinCosLogFlagListNew.append(sinCosLogFlagList[index])
            validFlagListNew.append(validFlagList[index])
            layerSymOpListNew.append(layerSymOpList[index])

        predList = predListNew
        probList = probListNew
        perLayerSymList = perLayerSymListNew
        sinCosLogFlagList = sinCosLogFlagListNew
        validFlagList = validFlagListNew
        layerSymOpList = layerSymOpListNew

    #确保候选的序列都是正常结束的
    predListNew = []
    probListNew = []
    for k in range(len(predList)):
        lenth = len(predList[k])
        if padToken == predList[k][lenth - 1]:
            predListNew.append(predList[k])
            probListNew.append(probList[k])
    predList = predListNew
    probList = probListNew

    return predList, probList

def test_BeamSearch_symSel(model, variConstNum, max_len1, device, IsTestInTrain, constValueTest, testData, labelTest1, labelTest2, maxCandNum):
    model.eval()
    with torch.no_grad():
        totalTestNum = testData.size(0)
        test_times = min(320, totalTestNum)
        count = 0
        #maxCandNum = 20
        notInTrainSucNum = 0
        testInTrainNum = 0
        max_len = max_len1  # + max_len2 + 2
        for i in range(test_times):
            isInTrain = IsTestInTrain[i, 0]
            if 0 < isInTrain:
                testInTrainNum = testInTrainNum + 1
            src = testData[i].unsqueeze(1).to(device)
            tgt = labelTest1[i].to(device)
            predList, probList = test_getSym(model, max_len1, maxCandNum, src, device)
            #predList, probList = test_getConnect(model, variConstNum, paramNums, max_len2, maxCandNum, src, predList, probList, device)
            success = 0
            k=-1
            for k in range(len(predList)):
                lenth = len(predList[k])
                if lenth < max_len + 1:
                    for j in range(max_len - lenth + 1):
                        predList[k].append(0)
                pred = torch.tensor(predList[k][1:max_len + 1]).to(device)
                if tgt.equal(pred):
                    if 0 == isInTrain:
                        notInTrainSucNum = notInTrainSucNum + 1
                        print("haha!!!")
                    count = count + 1
                    success = 1
                    break
            equal = GetEqual(tgt.cpu().numpy().tolist(), variConstNum, 6, labelTest2[i])
            constValueTmp = [float('{:.4f}'.format(i)) for i in constValueTest[i].numpy().tolist()]
            print(i, "isInTrain:", int(isInTrain), " success:", success, "hitNo:", k, " target:", tgt.cpu().numpy().tolist(),
                  " predict:", predList[0][1:max_len + 1], "equal:", equal, "const:", constValueTmp)
        print("testInTrainNum", testInTrainNum, " notInTrainSucNum:", notInTrainSucNum, "InTrain correct ratio:", (count - notInTrainSucNum) / testInTrainNum,
              "notInTrain correct ratio:", notInTrainSucNum / (test_times - testInTrainNum), "Predict correct ratio:", count / test_times)

def GetPredInit(labelTest1, index):
    predList = []
    probList = [1]
    predTmp = []
    lenth = labelTest1.size(1)
    hiddenNum = 0
    for i in range(lenth):
        if 0 == labelTest1[index, i]:
            break
        predTmp.append(int(labelTest1[index, i]))
        hiddenNum = hiddenNum + 1
    predTmp.append(0)
    predTmp.insert(0, 0)
    predList.append(predTmp)

    return predList, probList, hiddenNum

def test_BeamSearch_connect(model, paramNums, max_len2, device, IsTestInTrain, constValueTest, testData, labelTest1, labelTest2, maxCandNum, errorTh):
    model.eval()
    with torch.no_grad():
        totalTestNum = testData.size(0)
        test_times = min(320, totalTestNum)
        count = 0
        notInTrainSucNum = 0
        testInTrainNum = 0
        notInTrainError = 0
        failNum = 0
        approNum = 0
        approError = 0
        over1000Num = 0
        notInTrainLess1000Num = 0
        for i in range(test_times):
            isInTrain = IsTestInTrain[i, 0]
            if -1 == isInTrain:
                failNum = failNum + 1
                continue
            src = testData[i].unsqueeze(1).to(device)
            tgt = labelTest2[i].to(device)
            # predList, probList = test_getSym(model, max_len1, maxCandNum, src, device)
            predList, probList, hiddenNum = GetPredInit(labelTest1, i)
            max_len = max_len2 + hiddenNum + 1
            predList, probList = test_getConnect(model, variConstNum, paramNums, max_len2, maxCandNum, src, predList, probList, device)
            success = 0
            k = -1
            minTotalError = 0
            minConstList = []
            #predList, minConstList, minTotalError = GetFinalCand(variConstNum, symNetHiddenNum, testData[i].double(), constValueTest[i], predList)#这个函数只返回一个结果
            if 0 >= len(predList):
                failNum = failNum + 1
                continue
            if 0 < isInTrain:
                testInTrainNum = testInTrainNum + 1
            if 0 == isInTrain and minTotalError<errorTh:
                notInTrainError = notInTrainError + minTotalError
                notInTrainLess1000Num = notInTrainLess1000Num + 1
            for k in range(len(predList)):
                lenth = len(predList[k])
                if lenth < max_len+1:
                    for j in range(max_len+1 - lenth):
                        predList[k].append(padToken) #50
                pred = torch.tensor(predList[k][hiddenNum+2:max_len+1]).to(device)
                if tgt.equal(pred): #minTotalError < 0.00001: #tgt.equal(pred):
                    if 0 == isInTrain:
                        notInTrainSucNum = notInTrainSucNum + 1
                        print("haha!!!")
                    count = count + 1
                    success = 1
                    break
            if minTotalError >= errorTh:
                over1000Num = over1000Num + 1
            if 0==success and minTotalError<errorTh:
                approNum = approNum + 1
                approError = approError + minTotalError
            constValueTmp = [float('{:.4f}'.format(i)) for i in constValueTest[i].numpy().tolist()]
            print(i, "isInTrain:", int(isInTrain), " success:", success, "hitNo:", k, " target:", tgt.cpu().numpy().tolist(),
                  " predict:", predList[0][hiddenNum+2:max_len+1], "const:", constValueTmp)
            tgtEqual = GetEqual(labelTest1[i].cpu().numpy().tolist(), variConstNum, 6, labelTest2[i])
            preEqual = GetEqual(labelTest1[i].cpu().numpy().tolist(), variConstNum, 6, predList[0][hiddenNum+2:max_len+1])
            print("    tgtEqual:", tgtEqual, "preEqual:", preEqual)
            print("        求得的系数为:", minConstList[0:4], ", totalError:", minTotalError)
        test_times = test_times - failNum
        notInTrainError = notInTrainError / notInTrainLess1000Num
        approError = approError / approNum
        print("failNum:", failNum, "test_times:", test_times)
        print("testInTrainNum", testInTrainNum, " notInTrainSucNum:", notInTrainSucNum, "InTrain correct ratio:", (count-notInTrainSucNum)/testInTrainNum,
              "notInTrain correct ratio:", notInTrainSucNum/(test_times-testInTrainNum), "Predict correct ratio:", count/test_times)
        print("over1000Num:", over1000Num, "notInTrainLess1000Num:", notInTrainLess1000Num, "notInTrainError:", notInTrainError, "approNum:", approNum, "approError:", approError)


def test_BeamSearch_new(modelSym, model, paramNums, max_len1, max_len2, device, IsTestInTrain, constValueTest, testData, labelTest, maxCandNum1, maxCandNum2, errorTh, errorFlag):
    modelSym.eval()
    model.eval()
    errorList = []
    with torch.no_grad():
        totalTestNum = testData.size(0)
        test_times = min(320, totalTestNum)
        count = 0
        notInTrainSucNum = 0
        testInTrainNum = 0
        testInTrainError = 0
        notInTrainNum = 0
        notInTrainError = 0
        failNum = 0
        approNum = 0
        approError = 0
        over1000Num = 0

        max_len = max_len1 + max_len2 + 2
        for i in range(test_times):
            isInTrain = IsTestInTrain[i, 0]
            if -1==isInTrain:# or 53==i:# or 113==i or 192==i or 33==i:# or 164==i or 49==i or 223==i or 155==i or 256==i or 280==i:
                failNum = failNum + 1
                errorList.append(-1)
                continue
            src = testData[i].unsqueeze(1).to(device)
            tgt = labelTest[i].to(device)
            if modelSym is not None:
                predList, probList = test_getSym(modelSym, max_len1, maxCandNum1, src, device)
            else:
                predList, probList = test_getSym(model, max_len1, maxCandNum1, src, device)

            predListAll = []
            currNum = len(predList)
            for j in range(currNum):
                if 0 != predList[j][-1]:
                    continue
                predListTmp = []
                probListTmp = []
                predListTmp.append(predList[j])
                probListTmp.append(1)
                predListTmp, probListTmp = test_getConnect(model, variConstNum, paramNums, max_len2, maxCandNum2, src, predListTmp, probListTmp, device)
                for tmp in (predListTmp):
                    if padToken == tmp[-1]:
                        predListAll.append(tmp)
            predList = predListAll
            success = 0

            minTotalError = 0
            minConstList = []
            if 1 == errorFlag:
                predList, minConstList, minTotalError = GetFinalCand(variConstNum, symNetHiddenNum, testData[i].double(), constValueTest[i], predList)#这个函数只返回一个结果，误差打开1
            if 0 >= len(predList):
                failNum = failNum + 1
                errorList.append(-1)
                continue
            if 0 < isInTrain:
                testInTrainError = testInTrainError + minTotalError
                testInTrainNum = testInTrainNum + 1

            if 0 == isInTrain:# and minTotalError<errorTh:
                notInTrainError = notInTrainError + minTotalError
                notInTrainNum = notInTrainNum + 1

            for k in range(len(predList)):
                lenth = len(predList[k])
                if lenth < max_len + 1:
                    for j in range(max_len - lenth + 1):
                        predList[k].append(padToken) #50
                pred = torch.tensor(predList[k][1:max_len + 1]).to(device)
                if (1==errorFlag and minTotalError < 0.00001) or (0==errorFlag and tgt.equal(pred)): #error open 2
                #if tgt.equal(pred):
                    if 0 == isInTrain:
                        notInTrainSucNum = notInTrainSucNum + 1
                        print("haha!!!")
                    count = count + 1
                    success = 1
                    break
            if minTotalError >= errorTh:
                over1000Num = over1000Num + 1
            if 0==success: # and minTotalError<errorTh:
                approNum = approNum + 1
                approError = approError + minTotalError
            errorList.append(minTotalError)
            constValueTmp = [float('{:.4f}'.format(i)) for i in constValueTest[i].numpy().tolist()]
            print(i, "isInTrain:", int(isInTrain), " success:", success, "hitNo:", k, " target:", tgt.cpu().numpy().tolist(),
                  " predict:", predList[0][1:max_len + 1], "const:", constValueTmp)
            PrintTgtPreEqual(variConstNum, symNetHiddenNum, tgt, predList[0])
            print("        求得的系数为:", minConstList[0:4], ", totalError:", minTotalError)

        test_times = test_times - failNum
        if 0 < testInTrainNum:
            testInTrainError = testInTrainError / testInTrainNum
        if 0 < notInTrainNum:
            notInTrainError = notInTrainError / notInTrainNum
        if 0 < approNum:
            approError = approError / approNum
        notInTrainSucRatio = 0
        if 0 < (test_times - testInTrainNum):
            notInTrainSucRatio = notInTrainSucNum / (test_times - testInTrainNum)
        print("failNum:", failNum, "test_times:", test_times)
        if 0==testInTrainNum:
            testInTrainNum = 1
        print("testInTrainNum", testInTrainNum, "inTrainSucNum", count - notInTrainSucNum, "InTrain correct ratio:", (count - notInTrainSucNum) / testInTrainNum,
              " notInTrainSucNum:", notInTrainSucNum, "notInTrain correct ratio:", notInTrainSucRatio, "Predict correct ratio:", count / test_times)
        print("over1000Num:", over1000Num, "notInTrainNum:", notInTrainNum, "notInTrainError:",
              notInTrainError, "testInTrainError", testInTrainError, "approNum:", approNum, "approError:", approError)
    return errorList

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
        # 50 is replaced with padToken for compatibility with old data
        if 2 == variNum:
            for j in range(dim2):
                t = dim2 - 1 - j #Looking forward from behind
                if 50 == label2[i,t]:
                    label2[i, t] = padToken
                else:
                    break

        for j in range(dim1):
            if 0 == label1[i, j]:
                label[i, 0:j+1] = label1[i, 0:j+1]
                break
        len1 = j+1
        if 0 != label1[i, dim1-1]: #This situation indicates that label1 does not have 0, and therefore label is not assigned a value
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

def writeList(listALlErrorList, strName):
    f = open(strName, "w")
    f.write(str(listALlErrorList))
    f.close()

def readList(strName):
    f = open(strName, "r")
    listTmp = f.readline()
    listALlErrorList = eval(listTmp)
    f.close()
    return listALlErrorList

def PlotCurrInn(listError_quan, listError_two, IsTestInTrain, xlabel, ylabel, titleStr, flag):
    x = []
    y1 = []
    y2 = []
    y = []
    z = []
    num = 0
    for i in range(len(listError_quan)):
        if -1 == listError_quan[i] or -1 == listError_two[i]:
            continue
        if 0==flag and 0 == IsTestInTrain[i]:
            continue

        if 1==flag and 1 == IsTestInTrain[i]:
            continue

        if 2==flag:
            if 0.00001 > listError_quan[i] or 0.00001 > listError_two[i]:
                continue

        y1.append(listError_quan[i])
        y2.append(listError_two[i])
        y.append(listError_quan[i]-listError_two[i])
        z.append(0)
        x.append(num)
        num = num+1

    plt.figure(figsize=(6, 3.0))
    lw = 1.0

    plt.plot(x, y, color='darkorange', lw=lw, label='meanError difference')
    plt.plot(x, z, color='navy', lw=lw, label='Zero line')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(titleStr)
    plt.legend()#loc=locStr
    #plt.show()

def PlotCur(rltDir, flag):
    listError3_quan = readList(rltDir + "errorList1_3.txt")
    listError3_two = readList(rltDir + "errorList2_3.txt")
    IsTestInTrain3 = LoadIsTestInTrain("IsTestInTrain3")

    listError4_quan = readList(rltDir + "errorList1_4.txt")
    listError4_two = readList(rltDir + "errorList2_4.txt")
    IsTestInTrain4 = LoadIsTestInTrain("IsTestInTrain4")

    listError5_quan = readList(rltDir + "errorList1_5.txt")
    listError5_two = readList(rltDir + "errorList2_5.txt")
    IsTestInTrain5 = LoadIsTestInTrain("IsTestInTrain5")

    listError6_quan = readList(rltDir + "errorList1_6.txt")
    listError6_two = readList(rltDir + "errorList2_6.txt")
    IsTestInTrain6 = LoadIsTestInTrain("IsTestInTrain6")
    ############################################################################################################################################################

    #plt.figure(1)
    PlotCurrInn(listError3_quan, listError3_two, IsTestInTrain3, 'Index of expression', 'Difference of meanError', 'Test set with 3 hidden layers', flag)
    #plt.show()

    #plt.figure(2)
    PlotCurrInn(listError4_quan, listError4_two, IsTestInTrain4, 'Index of expression', 'Difference of meanError', 'Test set with 4 hidden layers', flag)
    #plt.show()

    #plt.figure(3)
    PlotCurrInn(listError5_quan, listError5_two, IsTestInTrain5, 'Index of expression', 'Difference of meanError', 'Test set with 5 hidden layers', flag)
    # plt.show()

    #plt.figure(4)
    PlotCurrInn(listError6_quan, listError6_two, IsTestInTrain6, 'Index of expression', 'Difference of meanError', 'Test set with 6 hidden layers', flag)
    # plt.show()

def GetRlt(errorList1, errorList2, IsTestInTrain, flag):
    s1 = 0
    s2 = 0
    total = 0
    for i in range(len(errorList1)):
        if -1 == errorList1[i] or -1 == errorList2[i]:
            continue

        if 0==flag and 0 == IsTestInTrain[i]:
            continue

        if 1==flag and 1 == IsTestInTrain[i]:
            continue
        if 2==flag:
            if 0.00001 > errorList1[i] or 0.00001 > errorList2[i]:
                continue

        if errorList1[i] <= errorList2[i]:
            s1 = s1+1
        if errorList2[i] <= errorList1[i]:
            s2 = s2 + 1
        total = total + 1
    return s1, s2, total

def GetAllRlt(rltDir, flag):

    listError3_quan = readList(rltDir+"errorList1_3.txt")
    listError3_two = readList(rltDir+"errorList2_3.txt")
    IsTestInTrain3 = LoadIsTestInTrain("IsTestInTrain3")

    listError4_quan = readList(rltDir+"errorList1_4.txt")
    listError4_two = readList(rltDir+"errorList2_4.txt")
    IsTestInTrain4 = LoadIsTestInTrain("IsTestInTrain4")

    listError5_quan = readList(rltDir+"errorList1_5.txt")
    listError5_two = readList(rltDir+"errorList2_5.txt")
    IsTestInTrain5 = LoadIsTestInTrain("IsTestInTrain5")

    listError6_quan = readList(rltDir+"errorList1_6.txt")
    listError6_two = readList(rltDir+"errorList2_6.txt")
    IsTestInTrain6 = LoadIsTestInTrain("IsTestInTrain6")

    s1, s2, total = GetRlt(listError3_quan, listError3_two, IsTestInTrain3, flag)
    print("s1, s2, total3:", s1, s2, total)

    s1, s2, total = GetRlt(listError4_quan, listError4_two, IsTestInTrain4, flag)
    print("s1, s2, total4:", s1, s2, total)

    s1, s2, total = GetRlt(listError5_quan, listError5_two, IsTestInTrain5, flag)
    print("s1, s2, total5:", s1, s2, total)

    s1, s2, total = GetRlt(listError6_quan, listError6_two, IsTestInTrain6, flag)
    print("s1, s2, total6:", s1, s2, total)

if __name__ == '__main__':
    with torch.no_grad():

        rltDir = "results/artificialData_rlt/"
        flag = 2
        GetAllRlt(rltDir, flag)
        PlotCur(rltDir, flag)

        paramNumList = [2, 2, 2, 2, 1, 1, 1, 1]  # , 1]
        symNum = len(paramNumList)
        paramNums = torch.tensor(paramNumList, dtype=torch.long)
        variNum = 3
        constNum = 1
        variConstNum = variNum + constNum
        symNetHiddenNum = 6
        maxLabelLen2 = 24 #12, 16, 20
        if 0 < HiddenNum:
            maxLabelLen2 = 12 + (HiddenNum-3)*4
        maxLabelLen = symNetHiddenNum + maxLabelLen2 + 2

        inputTest, labelTest1, labelTest2, constValueTest = LoadData("test3")
        labelTest = MergeLabel(maxLabelLen, labelTest1, labelTest2)
        IsTestInTrain = LoadIsTestInTrain("IsTestInTrain3")
        totalNumInTrain = IsTestInTrain.sum() # Number of test data in the training set

        testNum = labelTest1.size(0)
        print("test sample num:", testNum)
        histLenTest = GetLabelLen(labelTest1[0:320,:], symNetHiddenNum)
        print("histLenTest:", histLenTest)
        print("totalNumInTrain:", int(totalNumInTrain))

        ntoken = 257 #256+1
        d_model = 512
        nlayers = 6
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    modelSymSel_name = 'model_symSel/model_0.54989.pt'
    model_name = 'model/model_0.24265.pt' #0.24265

    modelSym = TransformerModel(variNum + 1, ntoken, d_model, nlayers=nlayers).to(device)
    LoadModel(modelSym, modelSymSel_name)
    modelTest = TransformerModel(variNum+1, ntoken, d_model, nlayers=nlayers).to(device)
    LoadModel(modelTest, model_name)

    test_BeamSearch_symSel(modelTest, variConstNum, symNetHiddenNum, device, IsTestInTrain, constValueTest, inputTest, labelTest1, labelTest2, 20)
    print('test_BeamSearch_symSel, 20 DSN1 finished')
    test_BeamSearch_symSel(modelSym, variConstNum, symNetHiddenNum, device, IsTestInTrain, constValueTest, inputTest, labelTest1, labelTest2, 20)
    print('test_BeamSearch_symSel, 20 DSN2 finished')

    test_BeamSearch_connect(modelTest, paramNums, maxLabelLen2, device, IsTestInTrain, constValueTest, inputTest, labelTest1, labelTest2, 2, 1000)
    print('test_BeamSearch_connect, 2,1000 finished')
    test_BeamSearch_connect(modelTest, paramNums, maxLabelLen2, device, IsTestInTrain, constValueTest, inputTest, labelTest1, labelTest2, 1, 1000)
    print('test_BeamSearch_connect, 1,1000 finished')


    test_BeamSearch_new(modelTest, modelTest, paramNums, symNetHiddenNum, maxLabelLen2, device, IsTestInTrain, constValueTest, inputTest, labelTest, 20, 1, 1000, 0)
    print('test_BeamSearch_new, 20 * 1 DSN1 new finished')
    test_BeamSearch_new(modelSym, modelTest, paramNums, symNetHiddenNum, maxLabelLen2, device, IsTestInTrain, constValueTest, inputTest, labelTest, 20, 1, 1000, 0)
    print('test_BeamSearch_new, 20 * 1 DSN2 new finished')

    errorList1 = test_BeamSearch_new(modelTest, modelTest, paramNums, symNetHiddenNum, maxLabelLen2, device, IsTestInTrain, constValueTest, inputTest, labelTest, 20, 1, 1000, 1)
    print('test_BeamSearch_new, 20 * 1 DSN1 new finished')
    errorList2 = test_BeamSearch_new(modelSym, modelTest, paramNums, symNetHiddenNum, maxLabelLen2, device, IsTestInTrain, constValueTest, inputTest, labelTest, 20, 1, 1000, 1)
    print('test_BeamSearch_new, 20 * 1 DSN2 new finished')

    writeList(errorList1, "results/artificialData_rlt/errorList1_3.txt")
    writeList(errorList2, "results/artificialData_rlt/errorList2_3.txt")
    s1, s2, total = GetRlt(errorList1, errorList2, IsTestInTrain, flag)
    print("s1, s2, total:", s1, s2, total)

    print('Finish Artificial Data Test!')