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
from numpy import *

def loadDataSet(fileName,delim='\t'):
    fr=open(fileName)
    stringArr=[line.strip().split(delim) for line in fr.readlines()]
    datArr=[list(map(float,line)) for  line in stringArr]
    return mat(datArr)

def pca(dataMat,topNfeat=9999999):
    meanVals=mean(dataMat,axis=0)
    meanRemoved=dataMat-meanVals
    covMat=cov(meanRemoved,rowvar=0)
    eigVals,eigVects=linalg.eig(mat(covMat))
    eigValInd=argsort(eigVals)
    eigValInd=eigValInd[:-(topNfeat+1):-1]
    redEigVects=eigVects[:,eigValInd]
    lowDDataMat=meanRemoved*redEigVects
    reconMat=(lowDDataMat*redEigVects.T)+meanVals
    return lowDDataMat,reconMat


def relplaceNanWithMean():
    dataMat=loadDataSet('secom.data',' ')
    numFeat=shape(dataMat)[1]
    for i in range(numFeat):
        meanVal=mean(dataMat[nonzero(~isnan(dataMat[:,i]))[0],i])
        dataMat[nonzero(isnan(dataMat[:,i].A))[0],i]=meanVal
    return dataMat




if __name__=="__main__":


    # dataMat=loadDataSet('testSet.txt')
    # lowDmat,reconMat=pca(dataMat,2)
    dataMat=relplaceNanWithMean()
    lowDmat, reconMat=pca(dataMat,10)
    # meanVals=mean(dataMat,axis=0)
    # meanRemoved=dataMat-meanVals
    # covMat=cov(meanRemoved,rowvar=0)
    # eigVals,eigVects=linalg.eig(mat(covMat))
    # print(eigVals)
