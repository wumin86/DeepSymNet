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
from sr_Transformer_generateData import  outPutSymRlt, GetComplexity, func
from scipy.optimize import least_squares, minimize
from gplearn import genetic
import time
import os
import sympy
from sympy.parsing.sympy_parser import parse_expr
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

padToken = 256

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
def GetRealExpress(p, codeStr):
    tmp = codeStr.replace("const", str(p[0]), 1)
    tmp = tmp.replace("const", str(p[1]), 1)
    tmp = tmp.replace("const", str(p[2]), 1)
    tmp = tmp.replace("const", str(p[3]), 1)
    tmp = tmp.replace("const", str(p[4]), 1)
    tmp = tmp.replace("const", str(p[5]), 1)
    tmp = tmp.replace("const", str(p[6]), 1)
    tmp = tmp.replace("const", str(p[7]), 1)
    tmp = tmp.replace("const", str(p[8]), 1)
    tmp = tmp.replace("const", str(p[9]), 1)

    return tmp

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
    for i in range(2): #求解一次就行了 2
        p0 = np.random.rand(10)
        #p0 = np.zeros(10)

        # constV = torch.from_numpy(p0)
        # flag = checkSymExpressValid(symNum, symNetHiddenNum, variConstNum, TestData, constV, pred)
        # if 0 == flag:
        #     break

        #fit_res = least_squares(error, p0, args=(data_x, data_y, codeStr))#,  # 将残差函数中的除p之外的参数都打包至args中
        fit_res = minimize(objFunc, p0, args=(data_x, data_y, codeStr), method='BFGS', options={'gtol': 1e-6, 'disp': False})
        # 拟合得到的结果是一个形如字典的 OptimizeResult 对象
        rlt = dict(fit_res)
        #totalError = sum((error(rlt["x"], data_x, data_y, codeStr)**2)) #均方误差
        totalError = sum(abs(error(rlt["x"], data_x, data_y, codeStr)))/20
        if totalError < minError:
            minError = totalError
            minConstList = rlt["x"]
        if minError < 1.0e-8:
            break

    #print("    求得的系数为:", minConstList, ", totalError:", minError)
    return minConstList, minError

def fit_func_test():
    """
    这里 p0 存放的是c1、c2的初始值，这个值会随着拟合的进行不断变化，使得误差
    error的值越来越小
    """

    #data_x = np.random.random((20,2))*4.0-2 #np.linspace(-1, 1, 10)
    data_x = np.random.rand(20, 2) * 4.0 - 2  # np.linspace(-1, 1, 10)

    codeStr = "b-(a*a)/(b+const)" # "a-const*a*b" # "a-np.cos(a+const)"
    data_y = func((-1.0748, -1.9766, 1.9207, -1.6063, 1.9207, -1.6063, 1.9207, -1.6063, -1, 1), data_x, codeStr)

    p0 = np.random.rand(8)
    #p0 = np.zeros(8)
    fit_res = least_squares(error, p0,
                                     args=(data_x, data_y, codeStr))#,  # 将残差函数中的除p之外的参数都打包至args中
                                     #bounds=((0, 0), (1000, np.inf)))  # 定义了两个边界值，0~1000和0~+∞。
    # 拟合得到的结果是一个形如字典的 OptimizeResult 对象
    rlt = dict(fit_res)
    totalError = sum((error(rlt["x"], data_x, data_y, codeStr) ** 2))
    print("求得的系数为:", rlt["x"], ", totalError:", totalError)
    print("haha")

def BFGS_test():
    """
    这里 p0 存放的是c1、c2的初始值，这个值会随着拟合的进行不断变化，使得误差
    error的值越来越小
    """

    data_x = np.random.rand(20, 2) * 4.0 - 2  # np.linspace(-1, 1, 10)
    #codeStr = "b-(a*a)/(b+const)" # "a-const*a*b" # "a-np.cos(a+const)"
    codeStr = "b-(a*a)/(b+const)"
    data_y = func((-1.0748, -1.9766, 1.9207, -1.6063, 1.9207, -1.6063, 1.9207, -1.6063, -1, 1), data_x, codeStr)

    p0 = np.random.rand(10)
    fit_res_ls = least_squares(error, p0, args=(data_x, data_y, codeStr))#,  # 将残差函数中的除p之外的参数都打包至args中
    rlt_ls = dict(fit_res_ls)
    totalError = sum((error(rlt_ls["x"], data_x, data_y, codeStr) ** 2))
    print("least_squares 求得的系数为:", rlt_ls["x"], ", totalError:", totalError)

    fit_res = minimize(objFunc, p0, args=(data_x, data_y, codeStr), method='BFGS', options={'gtol': 1e-6, 'disp': True})
    # 拟合得到的结果是一个形如字典的 OptimizeResult 对象
    rlt = dict(fit_res)
    totalError = sum((error(rlt["x"], data_x, data_y, codeStr) ** 2))
    print("BFGS 求得的系数为:", rlt["x"], ", totalError:", totalError)
    print("haha")

def GetConstValue(variConstNum, symNetHiddenNum, TestData, pred):
    symSelPred = [0 for i in range(symNetHiddenNum)]
    for i in range(len(pred)):
        if 0 == pred[i + 1]:
            break
        symSelPred[i] = pred[i + 1]
    predConnect = pred[i + 1 + 1:]
    codeStr = GetEqual(symSelPred, variConstNum, symNetHiddenNum, predConnect)
    a = codeStr.count('const')
    constList = []
    totalError = 9999
    if a <= 10:
        data_x = TestData[:, 0:variConstNum-1].numpy()
        data_y = TestData[:, variConstNum-1].numpy()
        constList, totalError = fit_func(data_x, data_y, codeStr, TestData, pred)
    return constList, totalError
#这个函数只返回一个结果
def GetFinalCand(variConstNum, symNetHiddenNum, TestData, allPredList):
    candNum = len(allPredList)
    minTotalError = 9999
    minConstList = []
    minIndex = -1
    finalCandList = []
    for i in range(candNum):
        if padToken != allPredList[i][-1]:
            continue
        # flag = checkSymExpressValid(symNum, symNetHiddenNum, variConstNum, TestData, constValueTest, allPredList[i])
        # if 0 == flag:
        #     continue
        constList, totalError = GetConstValue(variConstNum, symNetHiddenNum, TestData, allPredList[i])
        if totalError < minTotalError:
            minTotalError = totalError
            minConstList = constList
            minIndex = i
    if -1 != minIndex:
        finalCandList = [allPredList[minIndex]]
    return finalCandList, minConstList, minTotalError

#####################################################################################
def PrintTgtPreEqual(variConstNum, symNetHiddenNum, pred):
    symSelPred = [0 for i in range(symNetHiddenNum)]
    for i in range(len(pred)):
        if 0 == pred[i+1]:
            break
        symSelPred[i] = pred[i+1]
    predConnect = pred[i+1+1:]
    preEqual = GetEqual(symSelPred, variConstNum, symNetHiddenNum, predConnect)
    print("    preEqual:", preEqual)
    return preEqual

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
            continue #最后如果不是0的直接丢弃, 因为上面循环加1了
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
    preNodeNum = validFlag.size()[0] #上一层节点个数
    count = 0 #上一层有效节点个数
    for i in range(preNodeNum):
        if validFlag[i] == 1:
            count = count + 1
    if count == 0.0 or ((symNo == 1 or symNo == 3) and count == 1): #如果减号或除号且只有唯一有效节点，则不选
        return -1, None

    inp = torch.LongTensor(pred).unsqueeze(1).to(device)
    output = model(src, inp)
    outDim = output.size(2)
    outLen = output.size(0)

    output[outLen-1, 0, preNodeNum:outDim] = -100
    for i in range(preNodeNum):
        if 0 == validFlag[i]:
            output[outLen-1, 0, i] = -100

    numFlag = 0 #记录可用的sincoslog节点数
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
        # 如果是减号或者除号且两次选择节点一样，则不选
        if (symNo == 1 or symNo == 3) and 1 == opNo and preNode == i:
            output[outLen - 1, 0, i] = -100
            continue
        if symNo != 4 and symNo != 5 and symNo != 6 and symNo != 7:
            continue
        if sinCosLogFlag[i] == 0:
            continue
        output[outLen-1, 0, i] = -100

    norm_data = output[outLen - 1, 0, 0:preNodeNum] #wumin 只有这一部分有效
    norm_data = norm_data.unsqueeze(1)
    indexTmp = torch.zeros(preNodeNum, 2).to(device)
    for t in range(preNodeNum):
        indexTmp[t, 0] = candNo
        indexTmp[t, 1] = t
    outValues = torch.cat([indexTmp, norm_data], 1)
    outValues = outValues[outValues[:, 2].sort(descending=True)[1]]
    num = 0 #有效节点个数
    for i in range(preNodeNum):
        if 0 >= outValues[i, 2]:
            break
        num = num + 1
    if 0 == num:
        return -1, None
    outValues = outValues[0:num, :]
    outValues[:, 2] = (outValues[:, 2]/10.0) * prob
    #outValues = outValues[outValues[:, 2].sort(descending=True)[1]] #概率改变了需要重新排序 wumin 不需要啊,乘以的是同一个概率
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

    layerSymOp = copy.deepcopy(layerSymOpList[orgCandNo]) #wumin 注意深拷贝
    flag = 0 #0 没换层, 1换层了, 2 结束了
    if currSymNum-1 == symIndex and paramNum-1 == opNo: #此时要换层了
        if layerNum-1 == layerNo: #此时序列生成完毕
            flag = 2
        else: #此时要换层了
            flag = 1
            layerSymOp[0] = layerSymOp[0] + 1
            layerSymOp[1] = 0
            layerSymOp[2] = 0
            layerSymOp[3] = preNode
    else: #不用换层
        if paramNum-1 == opNo:#换新的运算符
            layerSymOp[1] = layerSymOp[1] + 1
            layerSymOp[2] = 0
        else: #只需要操作数换第二个
            layerSymOp[2] = layerSymOp[2] + 1
        layerSymOp[3] = preNode

    return flag, layerSymOp

#换行了, 更新 sinCosLogFlagListNew, validFlagListNew
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
    sinCosLogFlagList = [torch.zeros(variConstNum) for i in range(currTotalCandNum)] #换层时需要更新
    validFlagList = [torch.ones(variConstNum) for i in range(currTotalCandNum)] #换层时需要更新
    layerSymOpList = [] #每次需要更新
    for i in range(currTotalCandNum):
        layerSymOp = torch.zeros(4, dtype=int)
        layerSymOp[0] = 0 #层号
        layerSymOp[1] = 0 #运算符index
        layerSymOp[2] = 0 #第几个操作数
        layerSymOp[3] = -1 #存储上个节点编号preNode
        layerSymOpList.append(layerSymOp)
    # 最大长度应该计算出来!!!
    for i in range(max_len2):#这里的循环次数不用加1，因为如果预测结束了会额外加上padToken
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
            orgCandNo = int(outValueAll[t, 0])  # 原来所在的候选列表
            n = int(outValueAll[t, 1])  # 下一个元素
            newPred = predList[orgCandNo][:]
            newPred.append(n)
            probListNew.append(outValueAll[t, 2])
            perLayerSymListNew.append(perLayerSymList[orgCandNo])
            #layerSymOpListNew每次都要更新, 换层的时候根据选择的情况更新sinCosLogFlagListNew, validFlagListNew
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

def test_BeamSearch_new(modelSym, model, paramNums, max_len1, max_len2, device, testData, maxCandNum1, maxCandNum2):
    modelSym.eval()
    model.eval()
    with torch.no_grad():
        #totalTestNum = testData.size(0)
        #test_times = min(320, totalTestNum)#其实每次只有一个样本，test_times每次都为1
        max_len = max_len1 + max_len2 + 2
        minTotalError = 9999
        preEqual = ''
        #for i in range(test_times):
        src = testData[0].unsqueeze(1).to(device)
        predList, probList = test_getSym(modelSym, max_len1, maxCandNum1, src, device)

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
        predList, minConstList, minTotalError = GetFinalCand(variConstNum, symNetHiddenNum, testData[0].double(), predList)#这个函数只返回一个结果
        if 0 < len(predList):
            # 这里其实只有一个候选
            #for k in range(len(predList)):
            lenth = len(predList[0])
            if lenth < max_len + 1:
                for j in range(max_len - lenth + 1):
                    predList[0].append(padToken) #50
            if minTotalError < 0.00001:#tgt.equal(pred):
                success = 1
            print(" success:", success, "hitNo:", 0," predict:", predList[0][1:max_len + 1])
            preEqual = PrintTgtPreEqual(variConstNum, symNetHiddenNum, predList[0])
            print("        求得的系数为:", minConstList[0:4], ", totalError:", minTotalError)

    return minTotalError, preEqual, minConstList

def LoadModel(model, modelName):
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(modelName))
    else:
        model.load_state_dict(torch.load(modelName, map_location='cpu'))

def testGplearn(variNum, testDataIn):
    testData = testDataIn.squeeze(0)
    data_x = testData[:,0:variNum].numpy()
    data_y = testData[:, variNum].numpy()

    learner = genetic.SymbolicRegressor(verbose=1, population_size=2000, generations=20)
    # print("before y_pred:")
    # print(data_y)
    learner.fit(data_x, data_y)
    y_pred = learner.predict(data_x)
    # print("after y_pred:")
    # print(y_pred)
    num = testData.size(0)
    meanError = sum(abs(y_pred - data_y)) / num #20
    print("meanError:", meanError)

    express = learner._program
    express = str(express)
    converter = {
        'sub': lambda x, y: x - y,
        'div': lambda x, y: x / y,
        'mul': lambda x, y: x * y,
        'add': lambda x, y: x + y,
        'neg': lambda x: -x,
        'pow': lambda x, y: x ** y
    }
    express = sympy.sympify(express, locals=converter)
    complexity = GetComplexity(express)

    print("learner org:\n", learner._program)
    print("learner simplify:\n", express)
    print("complexity:", complexity)
    return meanError, complexity

def LoadBothMode(modelSym_name, model_name, ntoken):
    modelSym = TransformerModel(variNum + 1, ntoken, d_model, nlayers=nlayers).to(device)
    LoadModel(modelSym, modelSym_name)
    modelTest = TransformerModel(variNum + 1, ntoken, d_model, nlayers=nlayers).to(device)
    LoadModel(modelTest, model_name)
    return modelSym, modelTest


def TestAllData(cand1, cand2, modelList, allTestData):
    modelNum = len(modelList)
    modelNum = modelNum//2
    #expressNum = len(expressList)
    expressNum = allTestData.size(0)
    totalError = 0
    listALlError = []
    listALlComplexity = []
    for i in range(expressNum):
        listError = []
        listEqual = []
        listMinConst = []
        inputTest = allTestData[i]
        inputTest = inputTest.unsqueeze(0)
        for j in range(modelNum):
            print('modle No:', j)
            error, preEqual, minConst = test_BeamSearch_new(modelList[j*2], modelList[j*2+1], paramNums, symNetHiddenNum, maxLabelLen2, device, inputTest, cand1, cand2)
            listError.append(error)
            listEqual.append(preEqual)
            listMinConst.append(minConst)
        index = 0
        for j in range(len(listError) - 1):
            if listError[j + 1] < listError[index]:
                index = j + 1

        minError = listError[index]
        totalError = totalError + minError
        listALlError.append(minError)

        complexity = 9999
        preEqualMin = ""
        if listEqual[index] != "":
            preEqualMin = GetRealExpress(listMinConst[index], listEqual[index])
            preEqualMin = sympy.sympify(preEqualMin)
            complexity = GetComplexity(preEqualMin)
        listALlComplexity.append(complexity)

        print("expressNum:", i, "index", index, "minError:", minError, "complexity", complexity, "preEqualMin", preEqualMin)
    totalError = totalError/expressNum

    return totalError, listALlError, listALlComplexity

def TestAllGP(allTestData):
    expressNum = allTestData.size(0)
    totalError = 0
    listALlError = []
    listALlComplexity = []
    for i in range(expressNum):
        inputTest = allTestData[i]
        inputTest = inputTest.unsqueeze(0)
        error, complexity = testGplearn(variNum, inputTest)
        totalError = totalError + error
        listALlError.append(error)
        listALlComplexity.append(complexity)

    totalError = totalError/expressNum

    return totalError, listALlError, listALlComplexity

def MergeResultList(listALlError1, listALlComplexity1, listALlError2, listALlComplexity2):
    listALlError = []
    listALlComplexity = []
    num = len(listALlError1)
    for i in range(num):
        if listALlError1[i] <= listALlError2[i]:
            listALlError.append(listALlError1[i])
            listALlComplexity.append(listALlComplexity1[i])
        else:
            listALlError.append(listALlError2[i])
            listALlComplexity.append(listALlComplexity2[i])
    totalError = sum(listALlError)/num
    return totalError, listALlError, listALlComplexity

def LoadModelList(modelSym_name3, model_name3, modelSym_name4, model_name4, modelSym_name5, model_name5, modelSym_name6, model_name6):
    modelList = []

    modelSym3, modelTest3 = LoadBothMode(modelSym_name3, model_name3, ntoken)
    modelList.append(modelSym3)
    modelList.append(modelTest3)

    modelSym4, modelTest4 = LoadBothMode(modelSym_name4, model_name4, ntoken)
    modelList.append(modelSym4)
    modelList.append(modelTest4)

    modelSym5, modelTest5 = LoadBothMode(modelSym_name5, model_name5, ntoken)
    modelList.append(modelSym5)
    modelList.append(modelTest5)

    modelSym6, modelTest6 = LoadBothMode(modelSym_name6, model_name6, ntoken)
    modelList.append(modelSym6)
    modelList.append(modelTest6)

    return modelList


def writeList(totalErrorList, listALlErrorList, listALlComplexityList, timeList, strName):
    listAll = [totalErrorList, listALlErrorList, listALlComplexityList, timeList]
    f = open(strName, "w")
    f.write(str(listAll))
    f.close()
    haha = 0

def readList(strName):
    f = open(strName, "r")
    listTmp = f.readline()
    listAllTmp = eval(listTmp)
    f.close()
    totalErrorList = listAllTmp[0]
    listALlErrorList = listAllTmp[1]
    listALlComplexityList = listAllTmp[2]
    timeList = listAllTmp[3]

    return totalErrorList, listALlErrorList, listALlComplexityList, timeList

def GetScoreList(listCurrErrorList, listCurrComplexityList, listRealComplexity):
    algNum = len(listCurrErrorList)
    scoreList = []
    scorePairList_list = []
    for i in range(algNum):
        score = 0
        scorePairList = []
        for j in range(algNum):
            if j == i:
                scorePairList.append([-1, -1])
                continue
            expressNum = len(listCurrErrorList[0])
            s1=0
            s2=0
            for k in range(expressNum):
                #不是EQL则考虑复杂度
                #if 4 != i and 4 != j:
                if 0 != i and 0 != j:
                    if (listCurrErrorList[i][k]<=listCurrErrorList[j][k] or listCurrComplexityList[j][k] > 3 * listRealComplexity[k]) and listCurrComplexityList[i][k] <= 3 * listRealComplexity[k]:
                        s1 = s1+1
                    if (listCurrErrorList[j][k]<=listCurrErrorList[i][k] or listCurrComplexityList[i][k] > 3 * listRealComplexity[k]) and listCurrComplexityList[j][k] <= 3 * listRealComplexity[k]:
                        s2 = s2+1
                else:#EQL 不考虑复杂度
                    if listCurrErrorList[i][k]<=listCurrErrorList[j][k]:
                        s1 = s1+1
                    if listCurrErrorList[j][k]<=listCurrErrorList[i][k]:
                        s2 = s2+1
                
            if s1 >= s2:
                score = score + 1
            scorePairList.append([s1,s2])
        scoreList.append(score)
        scorePairList_list.append(scorePairList)

    return scoreList, scorePairList_list
#和随机算法比较，flag恒为0了
def PlotWholeCurTwo(listALlList_list, listALlComplexityList_list, listRealComplexityList):
    colorList=['darkorange', 'red', 'navy', 'green']
    labelList = ['full model', 'two model', 'gp', 'dsr']
    allTestDataNameList = ['Koza', 'Korns', 'Keijzer', 'Vlad','ODE', 'AIFeyman']
    algNum = len(listALlList_list)
    flag = 0
    yList = []
    for i in range(algNum):
        dataSetNum = len(listALlList_list[i])
        y = []
        for j in range(dataSetNum):
            yTmp = listALlList_list[i][j]
            for k in range(len(listALlList_list[i][j])):
                if 0==flag and listALlComplexityList_list[i][j][k]>3*listRealComplexityList[j][k]:
                    yTmp[k] = 100

                if 0==flag and 50<=yTmp[k]:
                    yTmp[k] = 100

                if 1==flag:
                    yTmp[k] = abs(yTmp[k]-listRealComplexityList[j][k])#abs
            y = y+yTmp
        yList.append(y)


    y = yList[0]
    z = []
    for i in range(len(y)):
        y[i] = y[i] - yList[1][i]
        if 10<=y[i]:
            y[i] = 10
        z.append(0)
    plt.figure(figsize=(6, 3.0))
    lw = 1.0
    x = range(len(y))
    plt.plot(x, y, color=colorList[0], lw=lw, label="meanError difference")
    plt.plot(x, z, color='navy', lw=lw, label='Zero line')

    s = 0
    for i in range(dataSetNum):
        num = len(listALlList_list[0][i])
        e = s + num
        m = (s+e)/2
        if 0 == flag:
            plt.vlines(e-1, -1, 0, color="blue", lw=0.5)  # 竖线
            plt.text(m-1, -1, "%s" % allTestDataNameList[i], fontsize=8, ha='center')
        else:
            plt.vlines(e - 1, -9, -2, color="blue", lw=0.5)  # 竖线
            plt.text(m, -9, "%s" % allTestDataNameList[i], ha='center')
        s=e

    plt.xlabel("Index of expression")
    if 0 == flag:
        plt.ylabel("meanError difference")
        plt.title("meaError difference on test set")
    else:
        plt.ylabel("Complexity difference")
        plt.title("Complexity difference on test set")
    # plt.title("titleStr")
    plt.legend()  # loc=locStr

#所有数据集画在一起
def PlotWholeCur(listALlList_list, listALlComplexityList_list, listRealComplexityList, flag):
    # colorList=['darkorange', 'red', 'blue', 'green', 'magenta'] #navy
    # labelList = ['DSN1', 'DSN2', 'GP', 'DSR', 'EQL']
    # 图示换个顺序
    #colorList = ['magenta', 'blue', 'green', 'darkorange', 'red']
    #colorList = ['lightsalmon', 'skyblue', 'yellowgreen', 'gold', 'plum']
    colorList = ['plum', 'gold', 'yellowgreen', 'skyblue', 'tomato']
    labelList = ['EQL', 'GP', 'DSR', 'DSN1', 'DSN2']

    allTestDataNameList = ['Koza', 'Korns', 'Keijzer', 'Vlad','ODE', 'AIFeyman']

    algNum = len(listALlList_list)
    plt.figure(figsize=(12, 4.5))
    lw = 1.0
    for i in range(algNum):
        dataSetNum = len(listALlList_list[i])
        y = []
        for j in range(dataSetNum):
            yTmp = listALlList_list[i][j]
            for k in range(len(listALlList_list[i][j])):
                if 0==flag and listALlComplexityList_list[i][j][k]>3*listRealComplexityList[j][k] and 0 !=i: #4 !=i:
                    yTmp[k] = 10 #max(yTmp[k],10) #10

                if 0==flag and 20 <= yTmp[k] and 0==i: #4==i:
                    yTmp[k] = 20 #max(yTmp[k],10) #10

                if 1==flag:
                    yTmp[k] = abs(yTmp[k]-listRealComplexityList[j][k])#abs
            y = y+yTmp
        x = range(len(y))
        plt.plot(x, y, color=colorList[i], lw=lw, label=labelList[i])

    s = 0
    for i in range(dataSetNum):
        num = len(listALlList_list[0][i])
        e = s + num
        m = (s+e)/2
        if 0 == flag:
            plt.vlines(e-1, -1.5, -0.2, color="blue", lw=0.5)  # 竖线  -1,-0.2
            plt.text(m, -1.5, "%s" % allTestDataNameList[i], ha='center')
        else:
            plt.vlines(e-1, -20, -5, color="blue", lw=0.5)  # 竖线 -9,-2
            plt.text(m, -20, "%s" % allTestDataNameList[i], ha='center')
        s=e

    plt.xlabel("Index of expression")
    if 0 == flag:
        plt.ylabel("meanError")
        plt.title("meaError on test set")
    else:
        plt.ylabel("Complexity difference")
        plt.title("Complexity difference on test set")
    # plt.title("titleStr")
    plt.legend()  # loc=locStr

def PlotCur(listALlList_list, listALlComplexityList_list, listRealComplexityList, flag):
    colorList=['darkorange', 'red', 'navy', 'green']
    labelList = ['full model', 'two model', 'gp', 'dsr']
    allTestDataNameList = ['Koza', 'Korns', 'Keijzer', 'Vladislavleva','ODE', 'AIFeyman']
    algNum = len(listALlList_list)
    dataSetNum = len(listALlList_list[0])
    for i in range(dataSetNum):
        plt.figure(figsize=(6, 3.0))
        lw = 0.5
        for j in range(algNum):
            y = listALlList_list[j][i]

            for k in range(len(y)):
                if 0==flag and listALlComplexityList_list[j][i][k]>3*listRealComplexityList[i][k]:
                    y[k] = 10
                if 1==flag:
                    y[k] = abs(y[k]-listRealComplexityList[i][k])
            x = range(len(y))
            plt.plot(x, y, color= colorList[j], lw=lw, label=labelList[j])

        plt.xlabel("Index of expression")
        if 0 == flag:
            plt.ylabel("meanError")
            plt.title("meaError on test set %s" %(allTestDataNameList[i]))
        else:
            plt.ylabel("Complexity difference")
            plt.title("Complexity difference on test set %s" % (allTestDataNameList[i]))
        #plt.title("titleStr")
        plt.legend()#loc=locStr
        #plt.show()

def PrintPairList(scorePairList_list):
    tmp=scorePairList_list
    if 4 == len(tmp):
        print("\hline")
        print("GP & %d:%d & %d:%d & %d:%d & %d:%d \\\\" %(tmp[0][0][0],tmp[0][0][1],tmp[0][1][0],tmp[0][1][1],tmp[0][2][0],tmp[0][2][1],tmp[0][3][0],tmp[0][3][1]))
        print("\hline")
        print("DSR & %d:%d & %d:%d & %d:%d & %d:%d \\\\" %(tmp[1][0][0],tmp[1][0][1],tmp[1][1][0],tmp[1][1][1],tmp[1][2][0],tmp[1][2][1],tmp[1][3][0],tmp[1][3][1]))
        print("\hline")
        print("DSN1 & %d:%d & %d:%d & %d:%d & %d:%d \\\\" %(tmp[2][0][0],tmp[2][0][1],tmp[2][1][0],tmp[2][1][1],tmp[2][2][0],tmp[2][2][1],tmp[2][3][0],tmp[2][3][1]))
        print("\hline")
        print("DSN2 & %d:%d & %d:%d & %d:%d & %d:%d \\\\" %(tmp[3][0][0],tmp[3][0][1],tmp[3][1][0],tmp[3][1][1],tmp[3][2][0],tmp[3][2][1],tmp[3][3][0],tmp[3][3][1]))
        print("\hline")
    if 5 == len(tmp):
        print("\hline")
        print("EQL & %d:%d & %d:%d & %d:%d & %d:%d & %d:%d \\\\" % (tmp[0][0][0], tmp[0][0][1], tmp[0][1][0], tmp[0][1][1], tmp[0][2][0], tmp[0][2][1], tmp[0][3][0], tmp[0][3][1], tmp[0][4][0], tmp[0][4][1]))
        print("\hline")
        print("GP & %d:%d & %d:%d & %d:%d & %d:%d & %d:%d \\\\" % (tmp[1][0][0], tmp[1][0][1], tmp[1][1][0], tmp[1][1][1], tmp[1][2][0], tmp[1][2][1], tmp[1][3][0], tmp[1][3][1], tmp[1][4][0], tmp[1][4][1]))
        print("\hline")
        print("DSR & %d:%d & %d:%d & %d:%d & %d:%d  & %d:%d\\\\" % (tmp[2][0][0], tmp[2][0][1], tmp[2][1][0], tmp[2][1][1], tmp[2][2][0], tmp[2][2][1], tmp[2][3][0], tmp[2][3][1], tmp[2][4][0], tmp[2][4][1]))
        print("\hline")
        print("DSN1 & %d:%d & %d:%d & %d:%d & %d:%d & %d:%d\\\\" % (tmp[3][0][0], tmp[3][0][1], tmp[3][1][0], tmp[3][1][1], tmp[3][2][0], tmp[3][2][1], tmp[3][3][0], tmp[3][3][1], tmp[3][4][0], tmp[3][4][1]))
        print("\hline")
        print("DSN2 & %d:%d & %d:%d & %d:%d & %d:%d & %d:%d\\\\" % (tmp[4][0][0], tmp[4][0][1], tmp[4][1][0], tmp[4][1][1], tmp[4][2][0], tmp[4][2][1], tmp[4][3][0], tmp[4][3][1], tmp[4][4][0], tmp[4][4][1]))
        print("\hline")

def GetALlResult():
    allTestDataNameList = ['allKozaTestData', 'allKornsTestData', 'allKeijzerTestData', 'allVladTestData',
                           'allODETestData', 'allAIFeymanTestData']

    listRealComplexityList = [[6, 5, 6, 4, 8, 10, 5, 5, 4, 4, 9], [2, 3, 4, 5, 7, 6], [4, 14, 5, 6, 7, 4, 5, 7, 6], [10, 14, 16, 4, 9, 12, 8], [8, 7, 6, 4, 3, 7, 6, 6, 5, 8, 9, 1],
    [7, 10, 1, 3, 1, 2, 3, 3, 3, 7, 2, 3, 11, 4, 2, 6, 3, 4, 9, 3, 3, 3, 4, 3, 4, 3, 3, 3, 3, 3, 3, 5, 5, 5, 3, 4]]
#上面这些内容不要改
    #####################################################################################################################################
    totalErrorList_quan, listALlErrorList_quan, listALlComplexityList_quan, timeList_quan = readList("results/public_rlt/DSN1Rlt.txt")
    totalErrorList_two, listALlErrorList_two, listALlComplexityList_two, timeList_two = readList("results/public_rlt/DSN2Rlt.txt")
    totalGPErrorList_gp, listALlGPErrorList_gp, listALlGPComplexityList_gp, timeList_gp = readList("results/public_rlt/gpRlt.txt")
    totalErrorList_dsr, listALlErrorList_dsr, listALlComplexityList_dsr, timeList_dsr = readList("results/public_rlt/dsrRlt.txt")
    totalErrorList_eql, listALlErrorList_eql, listALlComplexityList_eql, timeList_eql = readList("results/public_rlt/eqlRlt.txt")

    #图示换个顺序
    listALlErrorList_list = [listALlErrorList_eql, listALlGPErrorList_gp, listALlErrorList_dsr, listALlErrorList_quan, listALlErrorList_two]
    listALlComplexityList_list = [listALlComplexityList_eql, listALlGPComplexityList_gp, listALlComplexityList_dsr, listALlComplexityList_quan, listALlComplexityList_two]
    ######################################################################################################################################
    scoreListList = []
    for t in range(len(listALlErrorList_two)):#遍历每个数据集
        listCurrErrorList = []
        for i in range(len(listALlErrorList_list)):
            listCurrErrorList.append(listALlErrorList_list[i][t])

        listCurrComplexityList = []
        for i in range(len(listALlComplexityList_list)):
            listCurrComplexityList.append(listALlComplexityList_list[i][t])

        listRealComplexity = listRealComplexityList[t]

        scoreList, scorePairList_list = GetScoreList(listCurrErrorList, listCurrComplexityList, listRealComplexity)
        scoreListList.append(scoreList)
        print(allTestDataNameList[t], ":", scoreList)
        print(allTestDataNameList[t], ":", scorePairList_list)
        print(allTestDataNameList[t], ":")
        PrintPairList(scorePairList_list)
    print(scoreListList)

    #所有数据集画在一起
    PlotWholeCur(listALlErrorList_list, listALlComplexityList_list, listRealComplexityList, 0)
    PlotWholeCur(listALlComplexityList_list, listALlComplexityList_list, listRealComplexityList, 1)

def GetSemiRandCompareResult():

    listRealComplexityList = [[6, 5, 6, 4, 8, 10, 5, 5, 4, 4, 9], [2, 3, 4, 5, 7, 6], [4, 14, 5, 6, 7, 4, 5, 7, 6], [10, 14, 16, 4, 9, 12, 8], [8, 7, 6, 4, 3, 7, 6, 6, 5, 8, 9, 1],
    [7, 10, 1, 3, 1, 2, 3, 3, 3, 7, 2, 3, 11, 4, 2, 6, 3, 4, 9, 3, 3, 3, 4, 3, 4, 3, 3, 3, 3, 3, 3, 5, 5, 5, 3, 4]]
    #上面这些内容不要改
    ######################################################################################################################################
    totalErrorList_quan, listALlErrorList_quan, listALlComplexityList_quan, timeList_quan = readList("results/public_rlt/SemiRand/randRlt.txt")
    totalErrorList_two, listALlErrorList_two, listALlComplexityList_two, timeList_two = readList("results/public_rlt/SemiRand/DSN2Rlt.txt")

    listALlErrorList_list = [listALlErrorList_quan, listALlErrorList_two]
    listALlComplexityList_list = [listALlComplexityList_quan, listALlComplexityList_two]
    ######################################################################################################################################
    #和随机算法比较
    PlotWholeCurTwo(listALlErrorList_list, listALlComplexityList_list, listRealComplexityList)

def GetSemiRandModelList(modelListIn):
    modelList = []
    num = len(modelListIn)
    num = num//2
    for i in range(num):
        modelList.append(modelListIn[2*i])
        modelTestRand = TransformerModel(variNum + 1, ntoken, d_model, nlayers=nlayers).to(device)
        modelList.append(modelTestRand)

    return modelList

if __name__ == '__main__':
    with torch.no_grad():
        GetALlResult()
        GetSemiRandCompareResult()

        listAllComplexityList = [[6, 5, 6, 4, 8, 10, 5, 5, 4, 4, 9], [2, 3, 4, 5, 7, 6], [4, 14, 5, 6, 7, 4, 5, 7, 6],
                                  [10, 14, 16, 4, 9, 12, 8], [8, 7, 6, 4, 3, 7, 6, 6, 5, 8, 9, 1],
                                  [7, 10, 1, 3, 1, 2, 3, 3, 3, 7, 2, 3, 11, 4, 2, 6, 3, 4, 9, 3, 3, 3, 4, 3, 4, 3, 3, 3,
                                   3, 3, 3, 5, 5, 5, 3, 4]]
        paramNumList = [2, 2, 2, 2, 1, 1, 1, 1]  # , 1]
        symNum = len(paramNumList)
        paramNums = torch.tensor(paramNumList, dtype=torch.long)
        variNum = 3
        constNum = 1
        variConstNum = variNum + constNum
        symNetHiddenNum = 6
        maxLabelLen2 = 24 #12, 16, 20
        maxLabelLen = symNetHiddenNum + maxLabelLen2 + 2 #加2是因为有一个间隔符0和终止符 padToken
        inputSeqLen = 20 #输入序列长度

        ntoken = 257 #256
        d_model = 512#默认512   256
        nlayers = 6 #默认6       2
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if 3 == variNum:
        #3个变量的模型
        # 大数据充分训练的全模型
        modelSym_name3 = 'model/model_0.19990.pt'
        model_name3 = 'model/model_0.19990.pt'
        modelSym_name4 = 'model/model_0.27516.pt'
        model_name4 = 'model/model_0.27516.pt'
        modelSym_name5 = 'model/model_0.25057.pt'
        model_name5 = 'model/model_0.25057.pt'
        modelSym_name6 = 'model/model_0.22915.pt'
        model_name6 = 'model/model_0.22915.pt'

        modelList_DSN1 = LoadModelList(modelSym_name3, model_name3, modelSym_name4, model_name4, modelSym_name5, model_name5, modelSym_name6, model_name6)
        #modelList_DSN1 = GetSemiRandModelList(modelList_DSN1)

        # 大数据充分训练的模型
        modelSym_name3 = 'model_symSel/model_0.40559.pt'
        model_name3 = 'model/model_0.19990.pt'
        modelSym_name4 = 'model_symSel/model_0.62877.pt'
        model_name4 = 'model/model_0.27516.pt'
        modelSym_name5 = 'model_symSel/model_0.55540.pt'
        model_name5 = 'model/model_0.25057.pt'
        modelSym_name6 = 'model_symSel/model_0.60609.pt'
        model_name6 = 'model/model_0.22915.pt'
        modelList_DSN2 = LoadModelList(modelSym_name3, model_name3, modelSym_name4, model_name4, modelSym_name5, model_name5, modelSym_name6, model_name6)

    modelList = []

    modelSym3, modelTest3 = LoadBothMode(modelSym_name3, model_name3, ntoken)
    modelList.append(modelSym3)
    modelList.append(modelTest3)

    modelSym4, modelTest4 = LoadBothMode(modelSym_name4, model_name4, ntoken)
    modelList.append(modelSym4)
    modelList.append(modelTest4)

    modelSym5, modelTest5 = LoadBothMode(modelSym_name5, model_name5, ntoken)
    modelList.append(modelSym5)
    modelList.append(modelTest5)

    modelSym6, modelTest6 = LoadBothMode(modelSym_name6, model_name6, ntoken)
    modelList.append(modelSym6)
    modelList.append(modelTest6)

###########################################################################################

    if 3 == variNum:
        #3个变量
        allKozaTestData = torch.load('data/public_dataSet/Koza3.t')
        allKornsTestData = torch.load('data/public_dataSet/Korns3.t')
        allKeijzerTestData = torch.load('data/public_dataSet/Keijzer3.t')
        allVladTestData = torch.load('data/public_dataSet/Vlad3.t')
        allODETestData = torch.load('data/public_dataSet/ODE3.t')
        allAIFeymanTestData = torch.load('data/public_dataSet/AIFeyman3.t')

    cand1 = 10
    cand2 = 1

    allTestDataList = [allKozaTestData, allKornsTestData, allKeijzerTestData, allVladTestData, allODETestData, allAIFeymanTestData]
    allTestDataNameList = ['allKozaTestData', 'allKornsTestData', 'allKeijzerTestData', 'allVladTestData', 'allODETestData', 'allAIFeymanTestData']

    dataIndex = -1
    totalErrorList_quan = []
    listALlErrorList_quan = []
    listALlComplexityList_quan = []
    timeList_quan = []
    for i in range(len(allTestDataList)):
        if -1 != dataIndex:
            i = dataIndex
        print(allTestDataNameList[i])
        allTestData = allTestDataList[i]
        start = time.time()
        totalError_DSN1, listALlError_DSN1, listALlComplexity_DSN1 = TestAllData(cand1, cand2, modelList_DSN1, allTestData)
        end = time.time()
        time_quan = end - start
        timeList_quan.append(time_quan)
        totalErrorList_quan.append(totalError_DSN1)
        listALlErrorList_quan.append(listALlError_DSN1)
        listALlComplexityList_quan.append(listALlComplexity_DSN1)
        print("ours_DSN1", allTestDataNameList[i], "totalError:", totalError_DSN1, "listALlError", listALlError_DSN1, "listALlComplexity", listALlComplexity_DSN1, "run time(seconds)", end - start)
        if -1 != dataIndex:
            break
    writeList(totalErrorList_quan, listALlErrorList_quan, listALlComplexityList_quan, timeList_quan, "results/public_rlt/DSN1Rlt.txt")

    totalErrorList_two = []
    listALlErrorList_two = []
    listALlComplexityList_two = []
    timeList_two = []
    for i in range(len(allTestDataList)):
        if -1 != dataIndex:
            i = dataIndex
        print(allTestDataNameList[i])
        allTestData = allTestDataList[i]
        start = time.time()
        totalError_DSN2, listALlError_DSN2, listALlComplexity_DSN2 = TestAllData(cand1, cand2, modelList_DSN2, allTestData)
        end = time.time()
        time_two = end - start
        timeList_two.append(time_two)
        totalErrorList_two.append(totalError_DSN2)
        listALlErrorList_two.append(listALlError_DSN2)
        listALlComplexityList_two.append(listALlComplexity_DSN2)
        print("ours_DSN2", allTestDataNameList[i], "totalError:", totalError_DSN2, "listALlError", listALlError_DSN2, "listALlComplexity", listALlComplexity_DSN2, "run time(seconds)", end - start)
        if -1 != dataIndex:
            break
    writeList(totalErrorList_two, listALlErrorList_two, listALlComplexityList_two, timeList_two, "results/public_rlt/DSN2Rlt.txt")

    totalGPErrorList_gp = []
    listALlGPErrorList_gp = []
    listALlGPComplexityList_gp = []
    timeList_gp = []
    for i in range(len(allTestDataList)):
        if -1 != dataIndex:
            i = dataIndex
        print(allTestDataNameList[i])
        allTestData = allTestDataList[i]
        start = time.time()
        totalGPError_gp, listALlGPError_gp, listALlGPComplexity_gp = TestAllGP(allTestData)
        end = time.time()
        time_gp = end - start

        timeList_gp.append(time_gp)
        totalGPErrorList_gp.append(totalGPError_gp)
        listALlGPErrorList_gp.append(listALlGPError_gp)
        listALlGPComplexityList_gp.append(listALlGPComplexity_gp)
        print("GP", allTestDataNameList[i], "totalGPError:", totalGPError_gp, "listALlGPError:", listALlGPError_gp, "listALlGPComplexity:", listALlGPComplexity_gp, "run time(seconds):", end - start)
        if -1 != dataIndex:
            break
    writeList(totalGPErrorList_gp, listALlGPErrorList_gp, listALlGPComplexityList_gp, timeList_gp, "results/public_rlt/gpRlt.txt")

    print('Finish Public Data Test!')
    #GetALlResult()


