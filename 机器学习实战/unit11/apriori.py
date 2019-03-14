#!/usr/bin/env python
#-*- coding:utf-8 -*-

#----------------------------
#!  Copyright(C) 2018
#   All right reserved.
#   文件名称：
#   摘   要：
#   当前版本:
#   作   者：崔宗会
#   完成日期：
#-----------------------------
def loadDataSet():
    return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]

def createC1(dataSet):
    C1=[]
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return map(frozenset,C1)

def scanD(D,Ck,minSupport):
    ssCnt={}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if can not in ssCnt:
                    ssCnt[can]=1
                else:
                    ssCnt[can]+=1
    numItems=float(len(D))
    retList=[]
    supportData={}
    for key in ssCnt:
        support=ssCnt[key]/numItems
        if support>=minSupport:
            retList.insert(0,key)
        supportData[key]=support
    return retList,supportData

def aprioriGen(Lk,k):
    retList=[]
    lenLK=len(Lk)
    for i in range(lenLK):
        for j in range(i+1,lenLK):
            L1=list(Lk[i])[:k-2]
            L2=list(Lk[j])[:k-2]
            if L1==L2:
                retList.append(Lk[i]|Lk[j])
    return retList

def apriori(dataSet,minSupport=0.5):
    C1=list(createC1(dataSet))
    D=list(map(set,dataSet))
    L1,supportData=scanD(D,C1,minSupport)
    L=[L1]
    k=2
    sss=len(L[k-2])
    while (len(L[k-2])>0):
        Ck=aprioriGen(L[k-2],k)
        Lk,supK=scanD(D,Ck,minSupport)
        supportData.update(supK)
        L.append(Lk)
        k+=1
    return L,supportData


def generateRules(L,supportData,minConf=0.7):
    bigRuleList=[]
    for i in range(1,len(L)):
        for freqSet in L[i]:
            H1=[frozenset([item]) for item in freqSet]
            if (i<1):
                rulesFromConseq(freqSet,H1,supportData,bigRuleList,minConf)
            else:
                calcConf(freqSet,H1,supportData,bigRuleList,minConf)
    return bigRuleList
def calcConf(freqSet,H,supportData,br1,minConf=0.7):
    prunedH=[]
    for conseq in H:
        conf=supportData[freqSet]/supportData[freqSet-conseq]
        if conf >=minConf:
            print(freqSet-conseq,'-->',conseq,'conf:',conf)
            br1.append(freqSet-conseq,conseq,conf)
            prunedH.append(conseq)
    return prunedH


def rulesFromConseq(freqSet,H,supportData,br1,minConf=0.7):
    m=len(H[0])
    if (len(freqSet)>(m+1)):
        Hmp1=aprioriGen(H,m+1)
        Hmp1=calcConf(freqSet,Hmp1,supportData,br1,minConf)
        if (len(Hmp1)>1):
            rulesFromConseq(freqSet,Hmp1,supportData,br1,minConf)



if __name__=="__main__":
    dataSet=loadDataSet()
    # C1=list(createC1(dataSet))
    # D=list(map(set,dataSet))
    # L1,suppData0=scanD(D,C1,0.5)
    # DDD=0
    # L,suppData=apriori(dataSet)
    import numpy as np
    a0=np.array([2.5,0.5,2.2,1.9,3.1,2.3,2,1,1.5,1.1])

    a=np.array([0.69,-1.31,0.39,0.09,1.29,0.49,0.19,-0.81,-0.31,-0.71])
    b=np.array([0.49,-1.21,0.99,0.29,1.09,0.79,-0.31,-0.81,-0.31,-1.01])
    print(np.sum(a0 - np.mean(a0) - a)),
    # b=np.array([1,2])
    print(np.sum((a0 - np.mean(a0))*(a0 - np.mean(a0)))/(len(a)-1))
    print(np.sum(a*b)/(len(a)-1))
    print(np.sum(b*b)/(len(a)-1))
