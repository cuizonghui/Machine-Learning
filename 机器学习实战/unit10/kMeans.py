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
def loadDataSet(filename):
    dataMat=[]
    fr=open(filename)
    for line in fr.readlines():
        curLine=line.strip().split('\t')
        # ss=float(curLine)
        fltLine=list(map(float,curLine))

        dataMat.append(fltLine)
    return dataMat

def distEclud(vecA,vecB):
    return sqrt(sum(power(vecA-vecB,2)))

def randCent(dataSet,k):
    n=shape(dataSet)[1]
    centroids=mat(zeros((k,n)))
    for j in range(n):
        minJ=min(dataSet[:,j])
        rangeJ=float(max(dataSet[:,j])-minJ)
        centroids[:,j]=minJ+rangeJ*random.rand(k,1)
    return centroids



def kMeans(dataSet,k,distMeans=distEclud,createCent=randCent):
    m=shape(dataSet)[0]
    clusterAssment=mat(zeros((m,2)))
    centroids=createCent(dataSet,k)
    clusterChanged=True
    while clusterChanged:
        clusterChanged=False
        for i in range(m):
            minDist=inf
            minIndex=-1
            for j in range(k):
                distJI=distMeans(centroids[j,:],dataSet[i,:])
                if distJI<minDist:
                    minDist=distJI
                    minIndex=j
            if clusterAssment[i,0]!=minIndex:
                clusterChanged=True
            clusterAssment[i,:]=minIndex,minDist**2
        print(centroids)
        for cent in range(k):
            pstINClust=dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]
            centroids[cent,:]=mean(pstINClust,axis=0)
    return centroids,clusterAssment



def biKmeans(dataSet,k,distMeans=distEclud):
    m=shape(dataSet)[0]
    clusterAssment=mat(zeros((m,2)))
    centroid0=mean(dataSet,axis=0).tolist()[0]
    centList=[centroid0]
    for j in range(m):
        clusterAssment[j,1]=distMeans(mat(centroid0),dataSet[j,:])**2
    while (len(centList)<k):
        lowestSSE=inf
        for i in range(len(centList)):
            ptsInCurrCLuster=dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]
            centroidMat,splittClustAss=kMeans(ptsInCurrCLuster,k=k)
            sseSplit=sum(splittClustAss[:,1])
            sessNotSplit=sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0]])
            print("sseSplit,and not Split: ",sseSplit,sessNotSplit)
            if (sseSplit+sessNotSplit)<lowestSSE:
                bestCentToSplit=i
                bestNewCents=centroidMat
                bestClustAss=splittClustAss.copy()
                lowestSSE=sseSplit+sessNotSplit
        bestClustAss[nonzero(bestClustAss[:,0].A==1)[0],0]=len(centList)
        bestClustAss[nonzero(bestClustAss[:,0].A==1)[0],0]=bestCentToSplit
        print('the best CentToSplit is :',bestCentToSplit)
        print('the len of bestClustAss is :',len(bestClustAss))
        centList[bestCentToSplit]=bestNewCents[0,:]
        centList.append(bestCentToSplit[1,:])
        clusterAssment[nonzero(clusterAssment[:,0].A==\
                               bestCentToSplit)[0],:]=bestClustAss
    return mat(centList),clusterAssment




def distSLC(vecA,vecB):
    a=sin(vecA[0,1]*pi/180)*sin(vecB[0,1]*pi/180)
    b=cos(vecA[0,1]*pi/180)*cos(vecB[0,1]*pi/180)*cos(pi*(vecB[0,0]-vecA[0,0])/180)
    return arccos(a+b)*6371.0


import matplotlib
import matplotlib.pyplot as plt
def clusterClubs(numClust=5):
    datList=[]
    for line in open('places.txt').readlines():
        lineArr=line.split('\t')
        datList.append([float(lineArr[4]),float(lineArr[3])])
        datMat=mat(datList)
        myCentroids,clustAssing=biKmeans(datMat,numClust,distMeans=distSLC)

        fig=plt.figure()
        rect=[0.1,0.1,0.8,0.8]
        scatterMarkers=['s','o','^','8','p','d','v','h','>','<']
        axprops=dict(xticks=[],yticks=[])
        ax0=fig.add_axes(rect,label='ax0',**axprops)
        imgP=plt.imread('Portland.png')
        ax0.imshow(imgP)
        ax1=fig.add_axes(rect,label='ax1',frameon=False)
        for i in range(numClust):
            ptsInCurrCluster=datMat[nonzero(clusAssing[:,0].A==i)[0],:]
            markerStyle=scatterMarkers[i%len(scatterMarkers)]
            ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0],
                        ptsInCurrCluster[:,1].flatten().A[0],
                        marker=markerStyle,s=90)
        ax1.scatter(myCentroids[:,0].flatten().A[0],
                    myCentroids[:,1].flatten().A[0],marker='+',s=300)
        plt.show()



if __name__=='__main__':

    clusterClubs(5)

    # dataMat=mat(loadDataSet('testSet.txt'))
    # min_0=min(dataMat[:,0])
    # max_0=max(dataMat[:,0])
    # min_1=min(dataMat[:,1])
    # max_1=max(dataMat[:,1])
    # cen=randCent(dataMat,6)
    # # print(cen)
    #
    # myCEntroids,clusAssing=kMeans(dataMat,4)
    # print(myCEntroids)
    # print(clusAssing)
