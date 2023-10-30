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
import time
import sympy

padToken = 256 #256, 50
HiddenNum = -1 #-1混合训练, 大于0固定当前隐层数训练
#新的
def GetMCDisturb(variNum, symNo, symFlag, sinCosLogFlag, validFlag, paramNum):
    flagOut = 0
    if symFlag == 0:
        idxs = [-1, -1]
        tmp = torch.tensor(idxs, dtype=torch.long)
        tmp = tmp.unsqueeze(0)
        return tmp, flagOut

    num = validFlag.size()[0] #上一层节点个数
    indexNum = num

    count = 0 #上一层有效节点个数
    for i in range(indexNum):
        if validFlag[i] == 1:
            count = count + 1
    if count == 0.0 or ((symNo == 1 or symNo == 3) and count == 1): #如果减号或除号且只有唯一有效节点，则不选
        idxs = [-1, -1]
        tmp = torch.tensor(idxs, dtype=torch.long)
        tmp = tmp.unsqueeze(0)
        return tmp, flagOut

    prob = 1.0/count
    while True:
        randNum = torch.rand(1, 1)
        tmp1 = 0.0
        numFlag = 0
        for i in range(indexNum):
            if validFlag[i] == 0:
                continue
            if sinCosLogFlag[i] == 0:
                numFlag = numFlag + 1
            tmp1 = tmp1 + prob
            if tmp1 >= randNum:
                break
        index1 = int(i)
        if 1 == paramNum and indexNum-1 == index1: #一个操作数不能为常数
            continue
        if symNo != 4 and symNo != 5 and symNo != 6 and symNo != 7:
            flagOut = sinCosLogFlag[index1]
            break
        if numFlag == 0:
            flagOut = -1
            break
        if sinCosLogFlag[index1] == 0:
            flagOut = 1
            break
    if flagOut == -1:
        flagOut = 0
        idxs = [-1, -1]
        tmp = torch.tensor(idxs, dtype=torch.long)
        tmp = tmp.unsqueeze(0)
        return tmp, flagOut

    index2 = -1
    if paramNum == 2:
        while True:
            randNum = torch.rand(1, 1)
            tmp1 = 0.0
            for i in range(indexNum):
                if validFlag[i] == 0:
                    continue
                tmp1 = tmp1 + prob
                if tmp1 >= randNum:
                    break
            index2 = int(i)
            if index1 == index2 and indexNum-1 == index1:  #两个操作数不能都为常数
                continue
            #如果是减号或者除号且两次选择节点一样，则不选
            if (symNo == 1 or symNo == 3) and index1 == index2:
                continue
            else:
                break

    if -1 != index1 and -1 != index2:
        flagOut = sinCosLogFlag[index1] + sinCosLogFlag[index2]

    idxs = [index1, index2]
    tmp = torch.tensor(idxs, dtype=torch.long)
    tmp = tmp.unsqueeze(0)
    return tmp, flagOut


def GetAllMC(variNum, symValidFlag, sinCosLogFlag, validFlag, paramNums, preNodeNum):
    symNum = paramNums.size()[0]
    #preNodeNum = validFlag.size()[0]  # 上一层节点数
    currNodeNum = preNodeNum + symNum
    validFlagOut = torch.zeros(currNodeNum)
    sinCosLogFlagOut = torch.zeros(currNodeNum)

    allIndex, tmpFlag = GetMCDisturb(variNum, 0, symValidFlag[0], sinCosLogFlag, validFlag, paramNums[0])
    sinCosLogFlagOut[0] = tmpFlag
    if allIndex[0, 0] != -1:
        validFlagOut[0] = 1

    for i in range(symNum-1):
        tmpIndex, tmpFlag = GetMCDisturb(variNum, i+1, symValidFlag[i+1], sinCosLogFlag, validFlag, paramNums[i+1])
        sinCosLogFlagOut[i+1] = tmpFlag
        if tmpIndex[0, 0] != -1:
            validFlagOut[i+1] = 1
        allIndex = torch.cat([allIndex, tmpIndex], 0)

    for i in range(0, preNodeNum):
        validFlagOut[i+symNum] = validFlag[i]
        sinCosLogFlagOut[i+symNum] = sinCosLogFlag[i]

    return allIndex, sinCosLogFlagOut, validFlagOut


def GetNextLayerData(symNum, inputData, allIndex, validFlagIn):
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
        if i == 0:
            x[i] = (inputData[idx1] + inputData[idx2])

        if i == 1:
            x[i] = (inputData[idx1] - inputData[idx2])

        if i == 2:
            x[i] = (inputData[idx1] * inputData[idx2])

        if i == 3:
            if inputData[idx2] != 0:
                x[i] = (inputData[idx1] / inputData[idx2])
                if abs(x[i]) > 10000:
                    allIndex[i, 0] = -1
                    allIndex[i, 1] = -1
                    validFlagOut[i] = 0
            else:
                allIndex[i, 0] = -1
                allIndex[i, 1] = -1
                validFlagOut[i] = 0

        if i == 4:
            x[i] = torch.sin(inputData[idx1])

        if i == 5:
            x[i] = torch.cos(inputData[idx1])

        if i == 6:
            if inputData[idx1] < 10:
                x[i] = torch.exp(inputData[idx1])
            else:
                allIndex[i, 0] = -1
                allIndex[i, 1] = -1
                validFlagOut[i] = 0

        if i == 7:
            if inputData[idx1] > 0:
                x[i] = torch.log(inputData[idx1])
            else:
                allIndex[i, 0] = -1
                allIndex[i, 1] = -1
                validFlagOut[i] = 0
        if i == 8:
            x[i] = inputData[idx1]*inputData[idx1] #torch.pow(inputData[idx1], 2)

    for i in range(0, preNodeNum):
        x[i+symNum] = inputData[i]
        validFlagOut[i+symNum] = validFlagIn[i]

    return x, validFlagOut

def outPutSymRlt(variNum, symNum, hiddenNo, paramNums, indexSel, indexList, symFlag, symUseCountList):
    symNameList = ["+", "-", "*", "/", "sin", "cos", "exp", "log", "square"]
    #variNameList = ["a", "b", "const", "d", "e"]
    #variNameList = ["a", "b", "c", "const", "d", "e"]
    variNameListTmp = ["a", "b", "c", "d", "e"]
    variNameList = []
    for i in range(variNum):
        variNameList.append(variNameListTmp[i])
    variNameList.append("const")
    allIndex = indexList[hiddenNo]
    outStr = ""
    flag = 1
    if indexSel >= symNum:
        if hiddenNo > 0:
            outStr, flag = outPutSymRlt(variNum, symNum, hiddenNo-1, paramNums, indexSel-symNum, indexList, symFlag, symUseCountList)
        else:
            outStr = variNameList[indexSel - symNum] #print(variNameList[indexSel - symNum], end="")
            flag = 0
    else:
        if symFlag[indexSel] == 0:
            symUseCountList[indexSel] = symUseCountList[indexSel] + 1
            symFlag[indexSel] = 1
        if paramNums[indexSel] == 2:
            if hiddenNo > 0:
                tmp1, tmpFlag1 = outPutSymRlt(variNum, symNum, hiddenNo - 1, paramNums, allIndex[indexSel, 0], indexList, symFlag, symUseCountList)
                tmp2 = symNameList[indexSel]
                tmp3, tmpFlag2 = outPutSymRlt(variNum, symNum, hiddenNo - 1, paramNums, allIndex[indexSel, 1], indexList, symFlag, symUseCountList)
                if indexSel == 0:
                    outStr = tmp1 + tmp2 + tmp3

                if indexSel == 1:
                    outStr = tmp1 + tmp2
                    if tmpFlag2 == 1 or tmpFlag2 == 2:
                        outStr = outStr + "(" + tmp3 + ")"
                    else:
                        outStr = outStr + tmp3

                if indexSel == 2:
                    if tmpFlag1 == 0 or tmpFlag1 == 3:
                        outStr = tmp1 + tmp2
                    else:
                        outStr = "(" + tmp1 + ")" + tmp2
                    if tmpFlag2 == 0 or tmpFlag2 == 3:
                        outStr = outStr + tmp3
                    else:
                        outStr = outStr + "(" + tmp3 + ")"

                if indexSel == 3:
                    if tmpFlag1 == 0:
                        outStr = tmp1 + tmp2
                    else:
                        outStr = "(" + tmp1 + ")" + tmp2
                    if tmpFlag2 == 0:
                        outStr = outStr + tmp3
                    else:
                        outStr = outStr + "(" + tmp3 + ")"
            else:
                outStr = variNameList[allIndex[indexSel, 0]] + symNameList[indexSel] + variNameList[allIndex[indexSel, 1]]
            flag = indexSel + 1

        if paramNums[indexSel] == 1:
            if hiddenNo > 0:
                tmp, tmpFlag = outPutSymRlt(variNum, symNum, hiddenNo - 1, paramNums, allIndex[indexSel, 0], indexList, symFlag, symUseCountList)
                outStr = symNameList[indexSel] + "(" + tmp + ")"
            else:
                outStr = symNameList[indexSel] + "(" + variNameList[allIndex[indexSel, 0]] + ")"
            flag = 0
    return outStr, flag

#采样的节点变成紧凑型
def GetNewIndexList(maxLen2, labelListOrg, indexList):
    symTable  = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80]
    lenTable = len(symTable)
    lenth = len(labelListOrg)
    lisTmp = []

    tmpIndex = torch.zeros(lenTable, 2, dtype = int)
    tmpIndex[:] = -1
    newIndexList = [tmpIndex for i in range(lenth)]
    labelList2 = [torch.tensor(padToken) for i in range(maxLen2)]
    newFinal = -1
    num = 0
    lenLabel = 0
    for i in range(lenth):
        if maxLen2 < lenLabel:
            break
        if 0 == labelListOrg[i]:
            lisTmp.append(i)
        else:
            tmpIndex = torch.zeros(lenTable, 2, dtype = int)
            tmpIndex[:] = -1
            for j in range(lenTable):
                if 0 < labelListOrg[i]&symTable[j]:
                    newFinal = j
                    currIndex = indexList[i][j]
                    index1 = currIndex[0]
                    index2 = currIndex[1]
                    if 0 < len(lisTmp):
                        num1 = 0
                        num2 = 0
                        for k in range(len(lisTmp)):
                            tmp = lisTmp[k]
                            if index1 >= (i-tmp)*lenTable:
                                num1 = num1 + 1
                            if index2 >= (i - tmp) * lenTable:
                                num2 = num2 + 1
                        index1 = index1 - num1*lenTable
                        index2 = index2 - num2*lenTable
                    tmpIndex[j, 0] = index1
                    tmpIndex[j, 1] = index2
                    if lenLabel < maxLen2:
                        labelList2[lenLabel] = index1
                    lenLabel = lenLabel + 1
                    if maxLen2 < lenLabel:
                        newFinal = -1
                        break
                    if -1 != index2:
                        if lenLabel < maxLen2:
                            labelList2[lenLabel] = index2
                        lenLabel = lenLabel + 1
                        if maxLen2 < lenLabel:
                            newFinal = -1
                            break
            newIndexList[num] = tmpIndex
            num = num + 1
    if -1 != newFinal:
        newFinal = newFinal + (lenth-num)*lenTable
    return newIndexList, labelList2, newFinal

def GetLabel(variNum, maxLen2, symNetHiddenNum, symNum, indexFinalSel, indexList):
    nodeList = [indexFinalSel]
    nodeListList = [nodeList]
    for i in range(symNetHiddenNum):
        t = symNetHiddenNum-i-1 #从后往前
        allIndex = indexList[t]
        nodeListTmp = []
        for nodeNo in nodeList:
            if nodeNo < symNum:
                tmp0 = allIndex[nodeNo, 0]
                tmp1 = allIndex[nodeNo, 1]
                if not tmp0 in nodeListTmp:
                    nodeListTmp.append(int(tmp0))
                if tmp1 != -1:
                    if not tmp1 in nodeListTmp:
                        nodeListTmp.append(int(tmp1))
            else:
                a = nodeNo-symNum
                if not a in nodeListTmp:
                    nodeListTmp.append(a)
        nodeList = nodeListTmp
        nodeListList.append(nodeList)

    flag = 0
    lenth = len(nodeListList)
    for i in range(lenth):
        if 1 == flag:
            break
        t = symNetHiddenNum - i  # 从前往后
        for nodeNo in nodeListList[i]:
            if (t*symNum+variNum) == nodeNo: #必须包含常数项
                flag = 1
                break

    flag = 1 #可以不包含常数项
    labelListOrg = []
    labelList = [torch.tensor(0) for i in range(symNetHiddenNum)]
    numNot0 = 0
    for i in range(symNetHiddenNum):
        t = symNetHiddenNum - i - 1  # 变成从前往后了
        nodeList = nodeListList[t] #这一层的节点选择情况
        currLayerNum = len(nodeList)
        labelCurr = torch.tensor(0)
        for j in range(currLayerNum):
            if nodeList[j] < symNum:
                labelCurr = labelCurr + torch.pow(2, torch.tensor(nodeList[j]))
        labelListOrg.append(labelCurr)
        if 0 != labelCurr:
            labelList[numNot0] = labelCurr
            numNot0 = numNot0 + 1
    newIndexList = None
    labelList2 = None
    newFinal = -1
    #
    if (0 > HiddenNum or (3 == HiddenNum and HiddenNum >= numNot0) or (3 < HiddenNum and HiddenNum == numNot0)) and 1 == flag: #还是为了单独生成不同隐层数的表达式
        newIndexList, labelList2, newFinal = GetNewIndexList(maxLen2, labelListOrg, indexList) #采样的节点变成紧凑型

    return labelList, numNot0, newIndexList, labelList2, newFinal

def GetSeq2SeqData(sampleNumTh, variNum, maxLen2, symNum, inputDataShare, inputData, variConstNum, indexList, sampleNumHist):
    num = inputData.size()[0]  # 输入数据个数，即输入序列长度
    inputDim = inputData.size()[1]# 序列数据维度
    #validFlag = torch.ones(inputDim)
    symNetHiddenNum = len(indexList)  # 隐层个数
    outDataDim = variConstNum + symNetHiddenNum * symNum
    outData = torch.zeros(num, outDataDim)

    for i in range(num):
        outData[i, 0:variConstNum] = inputData[i, :]
        validFlag = torch.ones(inputDim)
        for j in range(symNetHiddenNum):  # 网络层数
            outData[i, :], validFlag = GetNextLayerData(symNum, outData[i, :], indexList[j], validFlag)

    shareDataNum = inputDataShare.size(0)
    outDataShare = torch.zeros(shareDataNum, outDataDim)
    for i in range(shareDataNum):
        outDataShare[i, 0:variConstNum] = inputDataShare[i, :]
        validFlagShare = torch.ones(inputDim)
        for j in range(symNetHiddenNum):  # 网络层数
            outDataShare[i, :], validFlagShare = GetNextLayerData(symNum, outDataShare[i, :], indexList[j], validFlagShare)

    inputSeqData = None
    labelSeq1 = None
    labelSeq2 = None
    indexLayerSeq = None
    finalIndexSeq = None
    outShareSeqData = None

    count = 0
    for i in range(outDataDim):
        if validFlag[i] == 1 and validFlagShare[i] == 1: #只要有一个数据非法就会造成validFlag[i]无效
            labelList1, numNot0, newIndexList, labelList2, newFinal = GetLabel(variNum, maxLen2, symNetHiddenNum, symNum, i, indexList)
            if (3 == HiddenNum and HiddenNum < numNot0) or (3 < HiddenNum and HiddenNum != numNot0) or -1 == newFinal:
                continue
            #if 100000 <= sampleNumHist[numNot0]:#采样不能太多
            if sampleNumTh <= sampleNumHist[numNot0]:  # 采样不能太多
                continue
            sampleNumHist[numNot0] = sampleNumHist[numNot0] + 1
            #常量表达式不要
            varTmp = torch.var(outDataShare[:, i])
            if varTmp < 0.00001:
                continue
            tmpOut = outData[:, i].unsqueeze(1)
            tmp = torch.cat([inputData, tmpOut], 1)
            tmp = tmp.unsqueeze(0)
            label1 = torch.LongTensor(labelList1)
            label1 = label1.unsqueeze(0)
            label2 = torch.LongTensor(labelList2)
            label2 = label2.unsqueeze(0)
            indexTmp = torch.stack(newIndexList).unsqueeze(0)
            selIndexTmp = torch.tensor(newFinal, dtype = int).unsqueeze(0)

            shareOutTmp = outDataShare[:, i] #.unsqueeze(1)
            shareOutTmp = shareOutTmp.unsqueeze(0)
            if 0 == count:
                inputSeqData = tmp
                labelSeq1 = label1
                labelSeq2 = label2
                indexLayerSeq = indexTmp
                finalIndexSeq = selIndexTmp
                outShareSeqData = shareOutTmp
            else:
                inputSeqData = torch.cat([inputSeqData, tmp], 0)
                labelSeq1 = torch.cat([labelSeq1, label1], 0)
                labelSeq2 = torch.cat([labelSeq2, label2], 0)
                indexLayerSeq = torch.cat([indexLayerSeq, indexTmp], 0)
                finalIndexSeq = torch.cat([finalIndexSeq, selIndexTmp], 0)
                outShareSeqData = torch.cat([outShareSeqData, shareOutTmp], 0)

            count = count + 1

    return outShareSeqData, inputSeqData, labelSeq1, labelSeq2, indexLayerSeq, finalIndexSeq, count

def GetInputSeq(num, variNum, constNum):
    variConstNum = variNum + constNum
    inputData = torch.rand(num, variConstNum) * 4.0 - 2
    for i in range(num):
        for j in range(constNum):
            inputData.data[i][j + variNum] = j + 1
    #下面进行排序
    for i in range(variNum):
        t = variNum - i - 1
        tmp = inputData[inputData[:, t].sort(stable=True)[1]]
        inputData = tmp
    return inputData

def GetProgress(processName, start, end, index, num):
    timePass = end-start
    ratio = (index+1)/num
    remnantTime = ((num-index-1)/(index+1))*timePass

    d = remnantTime//(3600*24)
    h = (remnantTime-d*3600*24)//3600
    m = (remnantTime-d*3600*24 -h*3600)/60
    print(processName+" finish ratio:", ratio, "remnantTime:", d, " day", h, " hour", m, " minute")

def GetSamples(sampleNum, sampleNumTh, maxLen2, inputSeqLen, variNum, constNum, symNum, paramNums, symNetHiddenNum):
    variConstNum = variNum + constNum
    symValidFlag = torch.ones(symNum)  # 用来表示运算符是否使用的标志
    sampleNumHist = torch.zeros(symNetHiddenNum+1)
    #symValidFlag[8:symNum] = 0
    inputSeqDatas = None
    labelSeq1s = None
    labelSeq2s = None
    indexLayerSeqs = None
    finalIndexSeqs = None
    outShareSeqDatas = None
    shareDataNum = 10
    inputDataShare = GetInputSeq(shareDataNum, variNum, constNum)  # 随机生成公共数据
    count = 0
    start = time.time()
    for n in range(sampleNum):
        flag = 1
        for i in range(symNetHiddenNum):
            #if 100000 > sampleNumHist[i+1]:
            if sampleNumTh > sampleNumHist[i + 1]:
                flag = 0
                break
        if 1 == flag:
            break
        validFlag = torch.ones(variConstNum)  # 输入数据都有效
        sinCosLogFlag = torch.zeros(variConstNum)
        nodeNum = variConstNum  # 第一层节点数(输入层)
        indexList = []
        for i in range(symNetHiddenNum):  # 网络隐层数目
            allIndex, sinCosLogFlag, validFlag = GetAllMC(variNum, symValidFlag, sinCosLogFlag, validFlag, paramNums, nodeNum) #对节点进行采样
            indexList.append(allIndex)
            nodeNum = nodeNum + symNum
        inputData = GetInputSeq(inputSeqLen, variNum, constNum) #随机生成数据
        outShareSeqData, inputSeqData, labelSeq1, labelSeq2, indexLayerSeq, finalIndexSeq, countValid = GetSeq2SeqData(sampleNumTh, variNum, maxLen2, symNum, inputDataShare, inputData, variConstNum, indexList, sampleNumHist)
        if 0 == countValid:
            continue
        if 0 == count:
            inputSeqDatas = inputSeqData
            labelSeq1s = labelSeq1
            labelSeq2s = labelSeq2
            indexLayerSeqs = indexLayerSeq
            finalIndexSeqs = finalIndexSeq
            outShareSeqDatas = outShareSeqData
        else:
            inputSeqDatas = torch.cat([inputSeqDatas, inputSeqData], 0)
            labelSeq1s = torch.cat([labelSeq1s, labelSeq1], 0)
            labelSeq2s = torch.cat([labelSeq2s, labelSeq2], 0)
            indexLayerSeqs = torch.cat([indexLayerSeqs, indexLayerSeq], 0)
            finalIndexSeqs = torch.cat([finalIndexSeqs, finalIndexSeq], 0)
            outShareSeqDatas = torch.cat([outShareSeqDatas, outShareSeqData], 0)
        count = count + 1
        end = time.time()
        GetProgress("GetSamples", start, end, n, sampleNum)
        print("n, sampleNum, sampleNumHist:", n, sampleNum, sampleNumHist)

    return outShareSeqDatas, inputSeqDatas, labelSeq1s, labelSeq2s, indexLayerSeqs, finalIndexSeqs

def PrintEqualLabelInfo(variNum, count, startId, label, currLabel, symNum, symNetHiddenNum, paramNums, finalIndexSeqs, indexLayerSeqs, labelIndex, currIndex):
    minSymFlag = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    symUseCountListTmp = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    labelStr = outPutSymRlt(variNum, symNum, symNetHiddenNum - 1, paramNums, finalIndexSeqs[labelIndex],
                            indexLayerSeqs[labelIndex], minSymFlag, symUseCountListTmp)
    labelStr = ''.join(labelStr[0])
    minSymFlag = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    symUseCountListTmp = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    currStr = outPutSymRlt(variNum, symNum, symNetHiddenNum - 1, paramNums, finalIndexSeqs[currIndex], indexLayerSeqs[currIndex],
                           minSymFlag, symUseCountListTmp)
    currStr = ''.join(currStr[0])
    print(count, "startId:", startId, "label:", label, "currLabel", currLabel, "labelStr:", labelStr,
          "currStr:", currStr)

def GetLabelSymNum(label):
    lenth = label.size(0)
    count = 0
    for i in range(lenth):
        for j in range(8):
            count = count + ((label[i]>>j)&1)
    return count

#选元素和最小的序列作为代表
def MergeEqualLabel(variNum, outShareSeqDatas, labelSeq1s, labelSeq2s, indexLayerSeqs, finalIndexSeqs, symNum, symNetHiddenNum, paramNums):
    expressNum = outShareSeqDatas.size(0) # 表达式个数
    outShareSeqLen = outShareSeqDatas.size(1)
    indexTmp = torch.zeros(expressNum, 1)
    labelFlag = torch.zeros(expressNum)
    for i in range(expressNum):
        indexTmp[i, 0] = i
    outValues = torch.cat([indexTmp, outShareSeqDatas], 1)
    # 排序
    for i in range(outShareSeqLen):
        t = outShareSeqLen - i
        outValues = outValues[outValues[:, t].sort(stable=True)[1]]
    count = 0
    startId = 0
    labelNum = 0
    labelIndexList = []
    start = time.time()
    for i in range(expressNum):
        dist = F.pairwise_distance(outValues[startId, 1:outShareSeqLen + 1].unsqueeze(0), outValues[i, 1:outShareSeqLen + 1].unsqueeze(0), p=2)  # pytorch求欧氏距离
        if dist < 0.0001:
        #if outValues[startId, 1:outShareSeqLen+1].equal(outValues[i, 1:outShareSeqLen+1]):
            if i < expressNum-1:
                continue
            else:
                i = i+1
        if i - startId > 1:
            labelIndex = int(outValues[startId, 0])
            label = labelSeq1s[labelIndex, :]
            labelSum = label.sum()
            labelSymNum = GetLabelSymNum(label)
            labelList = []
            for j in range(startId, i):
                currIndex = int(outValues[j, 0])
                currLabel = labelSeq1s[currIndex, :]
                currLabelSum = currLabel.sum()
                currLabelSymNum = GetLabelSymNum(currLabel)
                labelList.append(currIndex)
                labelFlag[currIndex] = labelNum
                if labelSymNum > currLabelSymNum or (labelSymNum == currLabelSymNum and labelSum > currLabelSum):
                    labelIndex = currIndex
                    label = currLabel
                    labelSum = currLabelSum
                    labelSymNum = currLabelSymNum
            for j in range(startId,i):
                currIndex = int(outValues[j, 0])
                if not labelSeq1s[currIndex, :].equal(label):
                    PrintEqualLabelInfo(variNum, count, startId, label, labelSeq1s[currIndex, :], symNum, symNetHiddenNum, paramNums, finalIndexSeqs, indexLayerSeqs, labelIndex, currIndex)
                    labelSeq1s[currIndex, :] = label
                    labelSeq2s[currIndex, :] = labelSeq2s[labelIndex, :]
                    count = count + 1
        else:
            labelIndex = int(outValues[startId, 0])
            labelList = [labelIndex]
            labelFlag[labelIndex] = labelNum
        labelIndexList.append(labelList)
        labelNum = labelNum + 1
        startId = i

        end = time.time()
        GetProgress("MergeEqualLabel", start, end, i, expressNum)

    return labelSeq1s, labelSeq2s, labelIndexList, labelFlag

def SaveData(inputSeqDatas, labelSeq1s, labelSeq2s, constValue, postfix):
    name1 = "data/inputSeqDatas-" + postfix + ".t"
    torch.save(inputSeqDatas, name1)

    name2 = "data/labelSeq1s-" + postfix + ".t"
    torch.save(labelSeq1s, name2)

    name2_1 = "data/labelSeq2s-" + postfix + ".t"
    torch.save(labelSeq2s, name2_1)

    name3 = "data/constValue-" + postfix + ".t"
    torch.save(constValue, name3)

def SaveIsTestInTrain(IsTestInTrain, postfix):
    #name2 = "data/IsTestInTrain.t"
    name2 = "data/" + postfix + ".t"
    torch.save(IsTestInTrain, name2)

def LoadData(postfix):
    name1 = "data/inputSeqDatas-" + postfix + ".t"
    inputSeqDatas = torch.load(name1)

    name2 = "data/labelSeq1s-" + postfix + ".t"
    labelSeq1s = torch.load(name2)

    name2_1 = "data/labelSeq2s-" + postfix + ".t"
    labelSeq2s = torch.load(name2_1)

    name3 = "data/constValue-" + postfix + ".t"
    constValue = torch.load(name3)

    return inputSeqDatas, labelSeq1s, labelSeq2s, constValue

def LoadIsTestInTrain(postfix):
    name2 = "data/" + postfix + ".t"
    IsTestInTrain = torch.load(name2)
    return IsTestInTrain

def GetNextLayerDataConst(symNum, inputData, allIndex, constNo, constValue, validFlagIn):
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
            if abs(x[i]) > 100000:
                allIndex[i, 0] = -1
                allIndex[i, 1] = -1
                validFlagOut[i] = 0
                print(x[i], " multi caution!!!")

        if i == 3:
            if inputData[idx2] != 0:
                x[i] = (inputData[idx1] / inputData[idx2])
                if abs(x[i]) > 10000:
                    allIndex[i, 0] = -1
                    allIndex[i, 1] = -1
                    validFlagOut[i] = 0
            else:
                allIndex[i, 0] = -1
                allIndex[i, 1] = -1
                validFlagOut[i] = 0

        if i == 4:
            x[i] = torch.sin(inputData[idx1])

        if i == 5:
            x[i] = torch.cos(inputData[idx1])

        if i == 6:
            if inputData[idx1] < 10:
                x[i] = torch.exp(inputData[idx1])
            else:
                allIndex[i, 0] = -1
                allIndex[i, 1] = -1
                validFlagOut[i] = 0

        if i == 7:
            if inputData[idx1] > 0:
                x[i] = torch.log(inputData[idx1])
                if abs(x[i])>100000:
                    allIndex[i, 0] = -1
                    allIndex[i, 1] = -1
                    validFlagOut[i] = 0
                    print(x[i]," log caution!!!")
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

def GetCurrIndex(symSel, connectSel, variConstNum, symNetHiddenNum):
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
                index2 = padToken
                if 2 == paramNums[j]:
                    index2 = connectSel[count]
                    count = count + 1
                if index2 < variConstNum + i * lenTable:
                    indexCurr[i, j, 1] = index2
                # else:
                #     indexCurr[i, j, 0] = -1
    finalIndex = lastSym + (symNetHiddenNum - lastLayer - 1) * lenTable
    return  indexCurr, finalIndex

def GetCurrSamples(symNum, variNum, constNum, symNetHiddenNum, inputSeqLen, maxConstLen, symSel, connectSel, num):
    variConstNum = variNum + constNum
    inputDim = variConstNum
    indexOrg, finalIndex = GetCurrIndex(symSel, connectSel, variConstNum, symNetHiddenNum)
    inputDatas = None
    constValues = None
    flag = 1
    maxCount = 5
    for n in range(num):
        if 0 == flag:
            break
        count = 0
        while True:
            count = count + 1
            indexCurr = indexOrg.clone()
            inputData = GetInputSeq(inputSeqLen, variNum, constNum)  # 随机生成数据
            constValue = torch.rand(maxConstLen) * 4.0 - 2
            for i in range(maxConstLen):
                if abs(constValue[i]) < 0.01:
                    constValue[i] = constValue[i] + 0.5

            outDataDim = variConstNum + symNetHiddenNum * symNum
            outData = torch.zeros(inputSeqLen, outDataDim)
            for i in range(inputSeqLen):
                outData[i, 0:variConstNum] = inputData[i, :]
                validFlag = torch.ones(inputDim)
                constNo = 0
                for j in range(symNetHiddenNum):  # 网络层数
                    outData[i, :], validFlag, constNo = GetNextLayerDataConst(symNum, outData[i, :], indexCurr[j], constNo, constValue, validFlag)
            if 0 == validFlag[finalIndex]:
                if count > maxCount:
                    flag = 0
                else:
                    continue
            tmpOut = outData[:, finalIndex]#.unsqueeze(1)
            inputData[:, variConstNum-1] = tmpOut
            inputData = inputData.unsqueeze(0)
            constValue = constValue.unsqueeze(0)
            break
        if 0 == n:
            inputDatas = inputData
            constValues = constValue
        else:
            inputDatas = torch.cat([inputDatas, inputData], 0)
            constValues = torch.cat([constValues, constValue], 0)
    return inputDatas, constValues, flag


def GetLabelLen1(labelData, symNetHiddenNum):
    histLen = [0 for i in range(symNetHiddenNum+1)]
    num = labelData.size(0)
    lenFlag = torch.zeros(num, 1, dtype=int)
    for i in range(num):
        tmp = labelData[i,:].nonzero()
        n = tmp.size(0)
        histLen[n] = histLen[n] + 1
        lenFlag[i] = n
    return histLen, lenFlag


##############################################################################
def GetClassLabelLen(labelIndexList, labelSeq1s, symNetHiddenNum):
    classNum = len(labelIndexList)
    hiddenNumTmp = torch.zeros(classNum, dtype=int)
    histLen = [0 for i in range(symNetHiddenNum+1)]
    for i in range(classNum):
        labelIndex = labelIndexList[i]
        symSel = labelSeq1s[labelIndex[0]]
        tmp = symSel.nonzero()
        n = tmp.size(0)
        histLen[n] = histLen[n] + 1
        hiddenNumTmp[i] = n
    return histLen, hiddenNumTmp
def SaveSepTestData(inputSeqDataTest, labelSeqTest1, labelSeqTest2, constValueTest, IsTestInTrain):
    totalNum = labelSeqTest1.size(0)
    inputSeqDataTestList = [None, None, None, None]
    labelSeqTest1List = [None, None, None, None]
    labelSeqTest2List = [None, None, None, None]
    constValueTestList = [None, None, None, None]
    IsTestInTrainList = [None, None, None, None]
    for i in range(totalNum):
        symSel = labelSeqTest1[i]
        tmp = symSel.nonzero()
        n = tmp.size(0)
        if 3 >= n:
            n = 0
        else:
            n = n-3
        if inputSeqDataTestList[n] is None:
            inputSeqDataTestList[n] = inputSeqDataTest[i].unsqueeze(0)
            labelSeqTest1List[n] = labelSeqTest1[i].unsqueeze(0)
            labelSeqTest2List[n] = labelSeqTest2[i].unsqueeze(0)
            constValueTestList[n] = constValueTest[i].unsqueeze(0)
            IsTestInTrainList[n] = IsTestInTrain[i].unsqueeze(0)
        else:
            inputSeqDataTestList[n] = torch.cat([inputSeqDataTestList[n], inputSeqDataTest[i].unsqueeze(0)], 0)
            labelSeqTest1List[n] = torch.cat([labelSeqTest1List[n], labelSeqTest1[i].unsqueeze(0)], 0)
            labelSeqTest2List[n] = torch.cat([labelSeqTest2List[n], labelSeqTest2[i].unsqueeze(0)], 0)
            constValueTestList[n] = torch.cat([constValueTestList[n], constValueTest[i].unsqueeze(0)], 0)
            IsTestInTrainList[n] = torch.cat([IsTestInTrainList[n], IsTestInTrain[i].unsqueeze(0)], 0)

    for i in range(4):
        SaveData(inputSeqDataTestList[i], labelSeqTest1List[i], labelSeqTest2List[i], constValueTestList[i], "test%d"%(i+3))
        SaveIsTestInTrain(IsTestInTrainList[i], "IsTestInTrain%d"%(i+3))

#测试集一半在训练集中出现过，一半没出现过，测试集每类只挑一个，全体合并, 论文第一部分实验生成数据
def CreatAllTrainDataNew(trainSampleNum, sampleNumTh, maxNumPerClass, maxLen2, paramNumList, inputSeqLen, variNum, constNum, symNetHiddenNum):
    symNum = len(paramNumList)
    paramNums = torch.tensor(paramNumList, dtype=torch.long)

    # torch.manual_seed(123)  # 初始化种子后每次运行的结果都一样
    outShareSeqDatas, inputSeqDatas, labelSeq1s, labelSeq2s, indexLayerSeqs, finalIndexSeqs = GetSamples(trainSampleNum, sampleNumTh, maxLen2, inputSeqLen, variNum, constNum,
                                                                                                         symNum, paramNums, symNetHiddenNum)

    labelSeq1s, labelSeq2s, labelIndexList, labelFlag = MergeEqualLabel(variNum, outShareSeqDatas, labelSeq1s, labelSeq2s, indexLayerSeqs, finalIndexSeqs, symNum, symNetHiddenNum, paramNums)

    hiddenNumHist, hiddenNumTmp = GetClassLabelLen(labelIndexList, labelSeq1s, symNetHiddenNum)

    validNum = 384
    testNum = 320
    semiTestNum = testNum//2
    trainNum = 250000
    trainClassNum = trainNum//maxNumPerClass
    classNum = len(labelIndexList)

    if validNum+semiTestNum > trainClassNum: #valid和test不能重合
        print("validNum+semiTestNum > trainClassNum!!!")
        return
    flag = 0

    if -1 == HiddenNum:
        for i in range(symNetHiddenNum-2):
            if 0 == i:
                currClassNum = hiddenNumHist[1] + hiddenNumHist[2] + hiddenNumHist[3]
            else:
                currClassNum = hiddenNumHist[i+3]
            if trainClassNum+semiTestNum > currClassNum:#测试集每类只挑一个，总数要够，即不在训练集的测试集要达到semiTestNum
                print(i+3, "   trainClassNum+semiTestNum > currClassNum!!!")
                flag = 1
    else:
        if 3 == HiddenNum:
            currClassNum = hiddenNumHist[1] + hiddenNumHist[2] + hiddenNumHist[3]
        else:
            currClassNum = hiddenNumHist[HiddenNum]
        if trainClassNum + semiTestNum > currClassNum:  # 测试集每类只挑一个，总数要够，即不在训练集的测试集要达到semiTestNum
            print(HiddenNum, "   trainClassNum+semiTestNum > currClassNum!!!")
            flag = 1

    if 1==flag:
        return

    inputSeqDataTrain = None
    constValueTrain = None
    inputSeqDataValid = None
    constValueValid = None
    inputSeqDataTest = None
    constValueTest = None

    labelSeqTrain1 = None
    labelSeqTrain2 = None
    labelSeqValid1 = None
    labelSeqValid2 = None
    labelSeqTest1 = None
    labelSeqTest2 = None

    IsTestInTrain = None
    currValidNum = 0
    savedHistLen = [0 for i in range(symNetHiddenNum + 1)]
    start = time.time()
    for i in range(classNum):
        labelIndex = labelIndexList[i]
        currHiddenNum = hiddenNumTmp[i]
        if 3 >= currHiddenNum:
            currHiddenNum = 3
        num1 = 0
        num2 = 0
        num3 = 0
        if savedHistLen[currHiddenNum] < trainClassNum:
            num1 = maxNumPerClass
        if savedHistLen[currHiddenNum] < validNum:
            num2 = 1
        if savedHistLen[currHiddenNum] >= validNum and savedHistLen[currHiddenNum] < validNum+semiTestNum:
            num3 = 1
        if savedHistLen[currHiddenNum] >= trainClassNum and savedHistLen[currHiddenNum] < trainClassNum+semiTestNum:
            num3 = 1

        num = num1 + num2 + num3
        symSel = labelSeq1s[labelIndex[0]]
        connectSel = labelSeq2s[labelIndex[0]]
        inputDatas, constValueTmp, flag = GetCurrSamples(symNum, variNum, constNum, symNetHiddenNum, inputSeqLen, maxLen2,
                                                   symSel, connectSel, num)
        if 0 == flag:
            continue
        savedHistLen[currHiddenNum] = savedHistLen[currHiddenNum] + 1
        size1 = symSel.size(0)
        size2 = connectSel.size(0)
        label1Tmp = torch.zeros(num, size1, dtype=int)
        label1Tmp[:, :] = symSel
        label2Tmp = torch.zeros(num, size2, dtype=int)
        label2Tmp[:, :] = connectSel

        if num1 > 0:
            if inputSeqDataTrain is None:
                inputSeqDataTrain = inputDatas[0:num1, :]
                constValueTrain = constValueTmp[0:num1, :]
                labelSeqTrain1 = label1Tmp[0:num1, :]
                labelSeqTrain2 = label2Tmp[0:num1, :]
            else:
                inputSeqDataTrain = torch.cat([inputSeqDataTrain, inputDatas[0:num1, :]], 0)
                constValueTrain = torch.cat([constValueTrain, constValueTmp[0:num1, :]], 0)
                labelSeqTrain1 = torch.cat([labelSeqTrain1, label1Tmp[0:num1, :]], 0)
                labelSeqTrain2 = torch.cat([labelSeqTrain2, label2Tmp[0:num1, :]], 0)

        if num2 > 0:
            if inputSeqDataValid is None:
                inputSeqDataValid = inputDatas[num1:num1 + num2, :]
                constValueValid = constValueTmp[num1:num1 + num2, :]
                labelSeqValid1 = label1Tmp[num1:num1 + num2, :]
                labelSeqValid2 = label2Tmp[num1:num1 + num2, :]
            else:
                inputSeqDataValid = torch.cat([inputSeqDataValid, inputDatas[num1:num1 + num2, :]], 0)
                constValueValid = torch.cat([constValueValid, constValueTmp[num1:num1 + num2, :]], 0)
                labelSeqValid1 = torch.cat([labelSeqValid1, label1Tmp[num1:num1 + num2, :]], 0)
                labelSeqValid2 = torch.cat([labelSeqValid2, label2Tmp[num1:num1 + num2, :]], 0)
            currValidNum = currValidNum + num2

        if num3 > 0:
            IsTestInTrainTmp = torch.zeros(num3, 1, dtype=int)
            if num1 > 0:
                IsTestInTrainTmp[:, 0] = 1
            if inputSeqDataTest is None:
                inputSeqDataTest = inputDatas[num1 + num2:num1 + num2 + num3, :]
                constValueTest = constValueTmp[num1 + num2:num1 + num2 + num3, :]
                labelSeqTest1 = label1Tmp[num1 + num2:num1 + num2 + num3, :]
                labelSeqTest2 = label2Tmp[num1 + num2:num1 + num2 + num3, :]
                IsTestInTrain = IsTestInTrainTmp
            else:
                inputSeqDataTest = torch.cat([inputSeqDataTest, inputDatas[num1 + num2:num1 + num2 + num3, :]], 0)
                constValueTest = torch.cat([constValueTest, constValueTmp[num1 + num2:num1 + num2 + num3, :]], 0)
                labelSeqTest1 = torch.cat([labelSeqTest1, label1Tmp[num1 + num2:num1 + num2 + num3, :]], 0)
                labelSeqTest2 = torch.cat([labelSeqTest2, label2Tmp[num1 + num2:num1 + num2 + num3, :]], 0)
                IsTestInTrain = torch.cat([IsTestInTrain, IsTestInTrainTmp], 0)
        end = time.time()
        GetProgress("GetCurrSamples", start, end, i, classNum)

    histLen1, lenFlag1 = GetLabelLen1(labelSeqTrain1, symNetHiddenNum)
    histLen2, lenFlag2 = GetLabelLen1(labelSeqValid1, symNetHiddenNum)
    histLen3, lenFlag3 = GetLabelLen1(labelSeqTest1, symNetHiddenNum)
    #print("totalNumInTrainOld: ", totalNumInTrainOld)
    a = inputSeqDataTrain[:, :, 2].max()  # 检查数据是否有非法的
    totalNumInTrain = IsTestInTrain.sum()  # 输出测试数据在训练集中的个数
    SaveData(inputSeqDataTrain, labelSeqTrain1, labelSeqTrain2, constValueTrain, "train")
    SaveData(inputSeqDataValid, labelSeqValid1, labelSeqValid2, constValueValid, "valid")
    SaveData(inputSeqDataTest, labelSeqTest1, labelSeqTest2, constValueTest, "test")
    SaveIsTestInTrain(IsTestInTrain, "IsTestInTrain")
    if -1 == HiddenNum:
        SaveSepTestData(inputSeqDataTest, labelSeqTest1, labelSeqTest2, constValueTest, IsTestInTrain)


#######################################################################################################################
#codeStr="const*np.sin(const-a)+np.cos(b)"
def func(p, x, codeStr):
    """
    Define the form of fitting function.
    """
    constTable = ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10"]
    c1, c2, c3, c4, c5, c6, c7, c8, c9, c10 = p

    if torch.is_tensor(x):
        variNum = x.size(1)
    else:
        variNum = x.shape[1]
    a = x[:, 0]
    b = x[:, 1]
    if 3 <= variNum:
        c = x[:, 2]
    if 4 <= variNum:
        d = x[:, 3]
    if 5 <= variNum:
        e = x[:, 4]

    tmp = codeStr.replace("const", constTable[0], 1)
    tmp = tmp.replace("const", constTable[1], 1)
    tmp = tmp.replace("const", constTable[2], 1)
    tmp = tmp.replace("const", constTable[3], 1)
    tmp = tmp.replace("const", constTable[4], 1)
    tmp = tmp.replace("const", constTable[5], 1)
    tmp = tmp.replace("const", constTable[6], 1)
    tmp = tmp.replace("const", constTable[7], 1)
    tmp = tmp.replace("const", constTable[8], 1)
    tmp = tmp.replace("const", constTable[9], 1)
    tmp = tmp.replace("sin", "np.sin", 20)
    tmp = tmp.replace("cos", "np.cos", 20)
    tmp = tmp.replace("exp", "np.exp", 20)
    tmp = tmp.replace("log", "np.log", 20)
    tmp = tmp.replace("sqrt", "np.sqrt", 20)
    outPut = eval(tmp)
    return outPut

def GetTestData(inputSeqLen, variNum, codeStr):
    data_x = torch.rand(inputSeqLen, variNum) * 4.0 - 2  #The same to train data set

    # 下面进行排序
    for i in range(variNum):
        t = variNum - i - 1
        tmp = data_x[data_x[:, t].sort(stable=True)[1]]
        data_x = tmp

    data_y = func((0, 0, 0, 0, 0, 0, 0, 0, 0, 0), data_x, codeStr)  # Because the codeStr here does not have const, the constant list of all zeros does not affect the results

    data_y = data_y.unsqueeze(1)
    testData = torch.cat([data_x, data_y], 1)
    testData = testData.unsqueeze(0)

    return testData

def GetALlTestData(inputSeqLen, variNum, expressList):
    allTestData = None
    num = len(expressList)
    for i in range(num):
        testData = GetTestData(inputSeqLen, variNum, expressList[i])
        if allTestData is None:
            allTestData = testData
        else:
            allTestData = torch.cat([allTestData, testData], 0)

    return allTestData

def GetComplexity(express):
    expressTmp = express
    for a in sympy.preorder_traversal(express):
        if isinstance(a, sympy.Float):
            expressTmp = expressTmp.subs(a, 0.5)

    expressTmp = str(expressTmp)
    complexity = 0
    complexity = complexity + expressTmp.count('+')
    complexity = complexity + expressTmp.count('-')
    complexity = complexity + expressTmp.count('*') - expressTmp.count('**')
    complexity = complexity + expressTmp.count('/')
    complexity = complexity + expressTmp.count('sin')
    complexity = complexity + expressTmp.count('cos')
    complexity = complexity + expressTmp.count('exp')
    complexity = complexity + expressTmp.count('log')
    complexity = complexity + expressTmp.count('sqrt')


    return complexity

def GetAllRealExpressComplexity(KozaList, KornsList, KeijzerList, VladList, ODEList, AIFeymanList):
    listKozaComplexity = []
    for symPress in (KozaList):
        symPress = sympy.sympify(symPress)
        clt = GetComplexity(symPress)
        listKozaComplexity.append(clt)

    listKornsComplexity = []
    for symPress in (KornsList):
        symPress = sympy.sympify(symPress)
        clt = GetComplexity(symPress)
        listKornsComplexity.append(clt)

    listKeijzerComplexity = []
    for symPress in (KeijzerList):
        symPress = sympy.sympify(symPress)
        clt = GetComplexity(symPress)
        listKeijzerComplexity.append(clt)

    listVladListComplexity = []
    for symPress in (VladList):
        symPress = sympy.sympify(symPress)
        clt = GetComplexity(symPress)
        listVladListComplexity.append(clt)

    listODEComplexity = []
    for symPress in (ODEList):
        symPress = sympy.sympify(symPress)
        clt = GetComplexity(symPress)
        listODEComplexity.append(clt)

    listAIFeymanComplexity = []
    for symPress in (AIFeymanList):
        symPress = sympy.sympify(symPress)
        clt = GetComplexity(symPress)
        listAIFeymanComplexity.append(clt)

    print("listKozaComplexity", listKozaComplexity)
    print("listKornsComplexity", listKornsComplexity)
    print("listKeijzerComplexity", listKeijzerComplexity)
    print("listVladListComplexity", listVladListComplexity)
    print("listODEComplexity", listODEComplexity)
    print("listAIFeymanComplexity", listAIFeymanComplexity)
    listAllComplexityList = [listKozaComplexity, listKornsComplexity, listKeijzerComplexity, listVladListComplexity, listODEComplexity, listAIFeymanComplexity]
    return listAllComplexityList

def CreatePublicTestData():
    with torch.no_grad():
        variNum = 3
        KozaList = ['a + a*a + a*a*a + a*a*a*a', 'a*a*a*a*a - 2*a*a*a + a', 'a*a*a*a*a*a - 2*a*a*a*a + a*a']
        NguyenList = ['a + a*a + a*a*a', 'a + a*a + a*a*a + a*a*a*a + a*a*a*a*a',
                      'a + a*a + a*a*a + a*a*a*a + a*a*a*a*a + a*a*a*a*a*a', 'sin(a*a)*cos(a)-1',
                      'sin(a)+sin(a+a*a)', 'sin(a)+sin(b*b)', '2*sin(a)*cos(b)']
        tmpStr = '1/(1+1/(a*a*a*a)) + 1/(1+1/(b*b*b*b))'
        KozaList = KozaList + NguyenList
        KozaList.append(tmpStr)

        KornsList = ['1.57+24.3*a', '-2.3+0.13*sin(a)',
                     '213.80940889*(1-exp(-0.54723748542*a))', '6.87+11*cos(7.23*a*a*a)',
                     '2-2.1*cos(9.8*a)*sin(1.3*b)']

        KeijzerList = ['0.3*a*sin(2*3.14159*a)', 'a*a*a*exp(-a)*cos(a)*sin(a)*(sin(a)*sin(a)*cos(a)-1)',
                       'log(a+sqrt(a*a+1))',
                       'a*b+sin((a-1)*(b-1))', 'a*a*a*a - a*a*a + b*b/2 - b', '6*sin(a)*cos(b)',
                       '8/(2+a*a+b*b)', 'a*a*a/5 + b*b*b/2 -b -a']

        VladList = ['exp(-(a-1)*(a-1))/(1.2+(b-2.5)*(b-2.5))', 'exp(-a)*a*a*a*cos(a)*sin(a)*(cos(a)*sin(a)*sin(a)-1)',
                    'exp(-a)*a*a*a*cos(a)*sin(a)*(cos(a)*sin(a)*sin(a)-1)*(b-5)', '6*sin(a)*cos(b)',
                    '(a-3)*(b-3)+2*sin((a-4)*(b-4))', '((a-3)**4+(b-3)**3-(b-3))/((b-2)**4 + 10)']

        ODEList = ['20-a-(a*b)/(1+0.5*a*a)', '10-(a*b)/(1+0.5*a*a)', '0.5*sin(a-b)-sin(a)', '-0.05*a*a-sin(b)',
                   'a-cos(b)/a',
                   '3*a-2*a*b-a*a', '2*b-a*b-b*b', 'a*(4-a-b/(1+a))', 'b*(a/(1+a)-0.075*b)',
                   '(cos(a)*cos(a)+0.1*sin(a)*sin(a))*sin(b)',
                   '10*(b-(1/3)*(a*a*a-a))', '-0.1*a']

        AIFeymanList = ['sqrt(2)*exp(-(a*a)/2)/(2*sqrt(3.14159))', 'sqrt(2)*exp(-(b*b)/(2*a*a))/(2*sqrt(3.14159)*a)',
                        'a*b',
                        'a*a*b/2', 'a/b', 'a*b/(2*3.14159)', '3*a*b/2', 'a/(4*3.14159*b*b)', 'a*b*b/2',
                        '(3*a*b)/(3-a*b)+1',
                        'a*b*b', 'a/(2*b+2)']

        if 3 == variNum:
            KornsList2 = ['0.23+14.2*(a+b)/(3*c)']
            KornsList = KornsList + KornsList2

            tmpStr3 = '(30*a*c)/((a-10)*b*b)'
            KeijzerList.append(tmpStr3)

            tmpStr3 = '30*((a-1)*(c-1))/(b*b*(a-10))'
            VladList.append(tmpStr3)


            AIFeymanList3 = ['sqrt(2)*exp(-((b-c)*(b-c))/(2*a*a))/(2*sqrt(3.14159)*a)',

                             'a/(4*3.14159*b*c*c)', 'a*b*c',
                             '(b+c)/(1+(b*c)/(a*a))', 'a*b*sin(c)', '1/(c/b+1/a)',
                             'a*(sin(b*c/2)*sin(b*c/2))/(sin(b/2)*sin(b/2))',
                             'c/(1-b/a)', '(b*c)/(a-1)',
                             'a/(4*3.14159*b*c)', '(3*a*a)/(20*3.14159*b*c)', 'a/(b*(c+1))',  #
                             '-a*b*cos(c)', 'a*b*c*c', '(a*b)/(2*3.14159*c)',
                             'a*b*c/2',
                             '(a*b)/(4*3.14159*c)', 'a*b*(c+1)', '(4*3.14159*a*b)/c',
                             'sin((2*3.14159*a*b)/c)*sin((2*3.14159*a*b)/c)',
                             '2*a*(1-cos(b*c))', '(a*a)/(8*3.14159*3.14159*b*c*c)', '(2*3.14159*a)/(b*c)',
                             'a*(b*cos(c)+1)']
            AIFeymanList3 = AIFeymanList + AIFeymanList3
            AIFeymanList = AIFeymanList3

        listAllComplexityList = GetAllRealExpressComplexity(KozaList, KornsList, KeijzerList, VladList, ODEList, AIFeymanList)  # Run once

        allKozaTestData = GetALlTestData(inputSeqLen, variNum, KozaList)
        torch.save(allKozaTestData, 'data/public_dataSet/Koza3.t')
        allKornsTestData = GetALlTestData(inputSeqLen, variNum, KornsList)
        torch.save(allKornsTestData, 'data/public_dataSet/Korns3.t')
        allKeijzerTestData = GetALlTestData(inputSeqLen, variNum, KeijzerList)
        torch.save(allKeijzerTestData, 'data/public_dataSet/Keijzer3.t')
        allVladTestData = GetALlTestData(inputSeqLen, variNum, VladList)
        torch.save(allVladTestData, 'data/public_dataSet/Vlad3.t')
        allODETestData = GetALlTestData(inputSeqLen, variNum, ODEList)
        torch.save(allODETestData, 'data/public_dataSet/ODE3.t')
        allAIFeymanTestData = GetALlTestData(inputSeqLen, variNum, AIFeymanList)
        torch.save(allAIFeymanTestData, 'data/public_dataSet/AIFeyman3.t')

if __name__ == '__main__':
    with torch.no_grad():
        trainSampleNum = 180000
        sampleNumTh = 100000
        maxNumPerClass = 20 # Number of samples generated per class
        paramNumList = [2, 2, 2, 2, 1, 1, 1, 1]  # , 1]
        symNum = len(paramNumList)
        paramNums = torch.tensor(paramNumList, dtype=torch.long)
        variNum = 3
        constNum = 1
        variConstNum = variNum + constNum
        symNetHiddenNum = 6
        maxLabelLen2 = 24  # 12, 16, 20
        if 0 < HiddenNum:
            maxLabelLen2 = 12 + (HiddenNum - 3) * 4
        maxLabelLen = symNetHiddenNum + maxLabelLen2 + 2  # Adding 2 is because there is a separator character 0 and a terminator character padToken
        inputSeqLen = 20  # Input sequence length

        CreatAllTrainDataNew(trainSampleNum, sampleNumTh, maxNumPerClass, maxLabelLen2, paramNumList, inputSeqLen, variNum, constNum, symNetHiddenNum)

        CreatePublicTestData()

        print('Finish Generate Training Data!')