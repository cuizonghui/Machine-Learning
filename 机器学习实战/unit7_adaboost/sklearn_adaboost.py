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
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles
from sklearn import svm
from sklearn.model_selection import  cross_val_score
x1,y1=make_gaussian_quantiles(cov=2.,n_samples=200,n_features=2,n_classes=2,
                              shuffle=True,random_state=1)


x2,y2=make_gaussian_quantiles(mean=(3,3),cov=1.5,n_samples=300,n_features=2,
                              n_classes=2,shuffle=True,random_state=1)

X=np.vstack((x1,x2))
y=np.hstack((y1,1-y2))

weakClassifier=DecisionTreeClassifier(max_depth=1)
weakClassifier_svm=svm.SVC()

clf=AdaBoostClassifier(base_estimator=weakClassifier,algorithm='SAMME',
                       n_estimators=30,learning_rate=0.8)

clf_svm=AdaBoostClassifier(base_estimator=weakClassifier_svm,algorithm='SAMME',


                           n_estimators=10,learning_rate=0.8)

clf.fit(X,y)
clf_svm.fit(X,y)
#绘制分类效果
x1_min=X[:,0].min()-1
x1_max=X[:,0].max()+1
x2_min=X[:,1].min()-1
x2_max=X[:,1].max()+1
x1_,x2_=np.meshgrid(np.arange(x1_min,x1_max,0.02),np.arange(x2_min,x2_max,0.02))

y_=clf.predict(np.c_[x1_.ravel(),x2_.ravel()])
y_=y_.reshape(x1_.shape)

y2_=(clf_svm.predict(np.c_[x1_.ravel(),x2_.ravel()])).reshape(x1_.shape)

scores=cross_val_score(clf,X,y)
print("决策树："+str(scores.mean())+"\n")

scores_svm=cross_val_score(clf_svm,X,y)
print("SVM："+str(scores_svm.mean())+"\n")


plt.figure(1,figsize=(6,12))
plt.subplot(211)

plt.contourf(x1_,x2_,y_,cmap=plt.cm.Paired)
plt.scatter(X[:,0],X[:,1],c=y)


plt.subplot(212)

plt.contourf(x1_,x2_,y2_,cmap=plt.cm.Paired)
plt.scatter(X[:,0],X[:,1],c=y)
plt.show()







# from sklearn.model_selection import  cross_val_score
# from sklearn.datasets import load_iris
# from sklearn.ensemble import AdaBoostClassifier
# iris=load_iris()
# clf=AdaBoostClassifier(n_estimators=100)
# scores=cross_val_score(clf,iris.data,iris.target)
# print(scores.mean())
