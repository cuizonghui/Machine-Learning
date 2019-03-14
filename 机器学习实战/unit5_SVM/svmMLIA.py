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
import numpy as np
from numpy import *
import matplotlib.pyplot as plt

import pymysql


def loadDataSet(filename):
    dataMat=[]
    labelMat=[]
    fr=open(filename)
    for line in fr.readlines():
        lineArr=line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

def selectJrand(i,m):
    j=i
    while (j==i):
        j=int(np.random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):
    if aj>H:
        aj=H
    if L>aj:
        aj=L
    return aj

def smoSimple(dataMatIn,classLabels,C,toler,maxIter):
    dataMatrix=np.mat(dataMatIn)
    labelMat=np.mat(classLabels).transpose()
    b=0
    m,n=np.shape(dataMatrix)
    alphas=np.mat(np.zeros((m,1)))
    iter=0
    while (iter<maxIter):
        alphaPairsChanged=0
        for i in range(m):
            fXi=float(np.multiply(alphas,labelMat).T*\
                      (dataMatrix*dataMatrix[i,:].T))+b
            Ei=fXi-float(labelMat[i])
            if ((labelMat[i]*Ei<-toler)and (alphas[i]<C))or \
                    ((labelMat[i]*Ei>toler)and (alphas[i]>0)):
                j=selectJrand(i,m)
                fXj=float(np.multiply(alphas,labelMat).T *\
                    (dataMatrix*dataMatrix[j,:].T))+b
                Ej=fXj-float(labelMat[j])
                alphaIold=alphas[i].copy()
                alphaJold=alphas[j].copy()
                if (labelMat[i]!=labelMat[j]):
                    L=max(0,alphas[j]-alphas[i])
                    H=min(C,C+alphas[j]-alphas[i])
                else:
                    L=max(0,alphas[j]+alphas[i]-C)
                    H=min(C,alphas[j]+alphas[i])
                if L==H:
                    print("L==H")
                    continue
                eta=2.0*dataMatrix[i,:]*dataMatrix[j,:].T- \
                    dataMatrix[i,:]*dataMatrix[i,:].T- \
                    dataMatrix[j,:]*dataMatrix[j,:].T
                if eta>=0:
                    print("eta>=0")
                    continue
                alphas[j]-=labelMat[j]*(Ei-Ej)/eta
                alphas[j]=clipAlpha(alphas[j],H,L)
                if (abs(alphas[j]-alphaJold)<0.00001):
                    print("j not moving enough")
                    continue
                alphas[i]+=labelMat[j]*labelMat[i]*(alphaJold-alphas[j])
                b1=b-Ei-labelMat[i]*(alphas[i]-alphaIold)*\
                    dataMatrix[i,:]*dataMatrix[i,:].T- \
                    labelMat[j]*(alphas[j]-alphaJold)* \
                    dataMatrix[i,:]*dataMatrix[j,:].T
                b2=b-Ej-labelMat[i]*(alphas[i]-alphaIold)*\
                    dataMatrix[i,:]*dataMatrix[j,:].T - \
                    labelMat[j]*(alphas[j]-alphaJold)* \
                    dataMatrix[j,:]*dataMatrix[j,:].T
                if (0<alphas[i]) and (C>alphas[i]):
                    b=b1
                elif(0<alphas[j]) and (C>alphas[j]):
                    b=b2
                else:
                    b=(b1+b2)/2.0
                alphaPairsChanged+=1
                print("iter:%d i:%d,pairs changed %d"%(iter,i,alphaPairsChanged
                    ))
        if(alphaPairsChanged==0):
            iter+=1
        else:
            iter=0
        print("iteration number: %d"%iter)
    return b,alphas



#
# def innerL(i,oS):
#     Ei=calcEk(oS,i)
#     if((oS.labelMat[i]*Ei<-oS.tol) and(oS.alphas[i]<oS.C)) or \
#             ((oS.labelMat[i]*Ei>oS.tol)and (oS.alphas[i]>0)):
#         j,Ej=selectJ(i,oS,Ei)
#         alphaIold=oS.alphas[i].copy()
#         alphaJold=oS.alphas[j].copy()
#         if(oS.labelMat[i]!=oS.labelMat[j]):
#             L=max(0,oS.alphas[j]-oS.alphas[i])
#             H=min(oS.C,oS.S+oS.alphas[j]+oS.alphas[i])
#         else:
#             L=max(0,oS.alphas[j]+oS.alphas[i]-oS.C)
#             H=min(oS.C,oS.alphas[j]+oS.alphas[i])
#         if L==H:
#             print("L==H")
#             return 0
#         eta=2.0*oS.X[i,:]*oS.X[j,:].T-oS.X[i,:]*oS.X[i,:].T -\
#             oS.X[j,:]*oS.X[j,:].T
#         if eta>=0:
#             print("eta>=0")
#             return 0
#         oS.alphas[j]=-oS.labelMat[j]*(Ei-Ej)/eta
#         oS.alphas[j]=clipAlpha(oS.alphas[j],H,L)
#         updateEk(oS,j)
#         if(abs(oS.alphas[j]-alphaJold)<0.00001):
#             print("j not moning enough")
#             return 0
#         updateEk(oS,i)
#         b1=oS.b-Ei-oS.labelMat[i]*(oS.alphas[i]-alphaIold)*\
#             oS.X[i,:]*oS.X[i,:].T-oS.labelMat[j]*\
#             (oS.alphas[j]-alphaJold)*oS.X[i,:]*oS.X[j,:].T
#         b2=oS.b-Ej-oS.labelMat[i]*(oS.alphas[i]-alphaIold)*\
#             oS.X[i,:]*oS.X[j,:].T-oS.labelMat[j]*\
#             (oS.alphas[j]-alphaJold)*(oS.X[j,:]*oS.X[j,:].T)
#         if(0<oS.alphas[i]) and (oS.C>oS.alphas[i]):
#             oS.b=b1
#         elif (0<oS.alphas[j]) and (oS.C >oS.alphas[j]):
#             oS.b=b2
#         else:
#             oS.b=(b1+b2)/2.0
#         return 1
#     else:return 0

# def innerL(i,oS):
#     Ei=calcEk(oS,i)
#     if((oS.labelMat[i]*Ei<-oS.tol) and(oS.alphas[i]<oS.C)) or \
#             ((oS.labelMat[i]*Ei>oS.tol)and (oS.alphas[i]>0)):
#         j,Ej=selectJ(i,oS,Ei)
#         alphaIold=oS.alphas[i].copy()
#         alphaJold=oS.alphas[j].copy()
#         if(oS.labelMat[i]!=oS.labelMat[j]):
#             L=max(0,oS.alphas[j]-oS.alphas[i])
#             H=min(oS.C,oS.S+oS.alphas[j]+oS.alphas[i])
#         else:
#             L=max(0,oS.alphas[j]+oS.alphas[i]-oS.C)
#             H=min(oS.C,oS.alphas[j]+oS.alphas[i])
#         if L==H:
#             print("L==H")
#             return 0
#         eta=2.0*oS.X[i,:]*oS.X[j,:].T-oS.X[i,:]*oS.X[i,:].T -\
#             oS.X[j,:]*oS.X[j,:].T
#         if eta>=0:
#             print("eta>=0")
#             return 0
#         oS.alphas[j]=-oS.labelMat[j]*(Ei-Ej)/eta
#         oS.alphas[j]=clipAlpha(oS.alphas[j],H,L)
#         updateEk(oS,j)
#         if(abs(oS.alphas[j]-alphaJold)<0.00001):
#             print("j not moning enough")
#             return 0
#         updateEk(oS,i)
#         b1=oS.b-Ei-oS.labelMat[i]*(oS.alphas[i]-alphaIold)*\
#             oS.X[i,:]*oS.X[i,:].T-oS.labelMat[j]*\
#             (oS.alphas[j]-alphaJold)*oS.X[i,:]*oS.X[j,:].T
#         b2=oS.b-Ej-oS.labelMat[i]*(oS.alphas[i]-alphaIold)*\
#             oS.X[i,:]*oS.X[j,:].T-oS.labelMat[j]*\
#             (oS.alphas[j]-alphaJold)*(oS.X[j,:]*oS.X[j,:].T)
#         if(0<oS.alphas[i]) and (oS.C>oS.alphas[i]):
#             oS.b=b1
#         elif (0<oS.alphas[j]) and (oS.C >oS.alphas[j]):
#             oS.b=b2
#         else:
#             oS.b=(b1+b2)/2.0
#         return 1
#     else:return 0


def calcWs(alphas,dataArr,classLbels):
    X=np.mat(dataArr)
    labelMat=np.mat(classLbels).transpose()
    m,n=shape(X)
    w=np.zeros((n,1))
    for i in range(m):
        w+=multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w


def plotfig_SVM(xMat,yMat,ws,b,alphas):
    xMat=np.mat(xMat)
    yMat=np.mat(yMat)
    b=np.array(b)[0]
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xMat[:,0].flatten().A[0],xMat[:,1].flatten().A[0])
    x=np.arange(-1.0,10.0,0.1)
    y=(-b-ws[0][0]*x)/ws[1][0]
    ax.plot(x,y)
    for i in range(100):
        if alphas[i]>0.0:
            ax.plot(xMat[i,0],xMat[i,1],'ro')
    plt.show()



def kernelTrans(X,A,kTup):
    m,n=shape(X)
    K=mat(zeros((m,1)))
    if kTup[0]=='lin':
        K=X*A.T
    elif kTup[0]=='rbf':
        for j in range(m):
            deltaRow=X[j,:]-A
            K[j]=deltaRow*deltaRow.T
        K=exp(K/(-1*kTup[1]**2))
    else:
        raise NameError("没有该核函数")
    return K

class optSTruct:
    def __init__(self,dataMatIn,classLabels,C,toler,kTup):
        self.X=dataMatIn
        self.labelMat=classLabels
        self.C=C
        self.tol=toler
        self.m=shape(dataMatIn)[0]
        self.alphas=mat(zeros((self.m,1)))
        self.b=0
        self.eCache=mat(zeros(self.m,2))
        self.K=mat(zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i]=kernelTrans(self.X,self.X[i,:],kTup)







if __name__=='__main__':
    dataArr,labelArr=loadDataSet('testSet.txt')
    b,alphas=smoSimple(dataArr,labelArr,0.6,0.001,40)

    ws=calcWs(alphas,dataArr,labelArr)
    plotfig_SVM(dataArr,labelArr,ws,b,alphas)



    print(b)
    print(alphas)


    dd=np.mat(labelArr)
    dfdf=0
