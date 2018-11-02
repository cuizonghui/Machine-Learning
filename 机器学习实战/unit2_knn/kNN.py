#!/usr/bin/env python
#-*- coding:utf-8 -*-

#----------------------------
#!  Copyright(C) 2018, 北京博众
#   All right reserved.
#   文件名称：
#   摘   要：
#   当前版本:
#   作   者：崔宗会
#   完成日期：
#-----------------------------
import numpy as np
import operator

def createDataSet():
    group=np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels



def classify0(inX,dataSet,labels,k):
    """
    k近邻域算法
    :param inX: 输入向量
    :param dataSet: 训练数据
    :param labels: 标签
    :param k: ｋ
    :return: 分类结果
    """
    dataSetSize=dataSet.shape[0]
    diffMat=np.tile(inX,(dataSetSize,1))-dataSet ###tile是沿轴复制数据
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(axis=1)
    distances=sqDistances**0.5 ##最小二乘距离
    sortedDistIndicies=distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    sortedClassCount=sorted(classCount.items(),
                            key=operator.itemgetter(1),reverse=True)
    # sortedClassCount = sorted(classCount.items(),
    #                           key=lambda d:d[1], reverse=True)

    return sortedClassCount[0][0]


def file2matrix(filaname):
    fr=open(filaname)
    arrayOLines=fr.readlines()
    numberOfLines=len(arrayOLines)
    returnMat=np.zeros((numberOfLines,3))
    classLabelVector=[]
    index=0
    for line in arrayOLines:
        line=line.strip()  ##去掉回车
        listFromLine=line.split('\t')##以\t分割
        returnMat[index,:]=listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index+=1
    return returnMat,classLabelVector


def autoNorm(dataSet):
    minVals=dataSet.min(0)
    maxVals=dataSet.max(0)
    ranges=maxVals-minVals
    normDataSet=np.zeros(np.shape(dataSet))
    m=dataSet.shape[0]
    normDataSet=dataSet-np.tile(minVals,(m,1))
    normDataSet=normDataSet/np.tile(ranges,(m,1))
    return normDataSet,ranges,minVals



def datingClassTest():
    hoRatio=0.10
    datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals=autoNorm(datingDataMat)
    m=normMat.shape[0]
    numTestVecs=int(m*hoRatio)
    errorCount=0.0
    for i in range(numTestVecs):
        classifierResult=classify0(normMat[i,:],normMat[numTestVecs:m,:],\
                                   datingLabels[numTestVecs:m],3)
        print("the classifier came back with: %d, the real answer is : %d"\
              %(classifierResult,datingLabels[i]))
        if (classifierResult!=datingLabels[i]):
            errorCount+=1
    print("the total error rate is: %f"%(errorCount/float(numTestVecs)))





if __name__=='__main__':
    import matplotlib.pyplot as plt
    datingClassTest()
    # datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')
    # normMat,ranges,minVals=autoNorm(datingDataMat)
    #
    # print(ranges)
    #
    #
    # fig=plt.figure()
    # ax=fig.add_subplot(111)
    # ax.scatter(datingDataMat[:,1],datingDataMat[:,2],
    #            15.0*np.array(datingLabels),15.0*np.array(datingLabels))
    # plt.show()


    #
    #
    #
    #
    # group,labels=createDataSet()
    # predict=classify0([0,0],group,labels,3)
    # ddd=0