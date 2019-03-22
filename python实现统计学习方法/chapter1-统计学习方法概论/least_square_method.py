# ----------------------------
#!  Copyright(C) 2019
#   All right reserved.
#   文件名称：xxx.py
#   摘   要：xxx
#   当前版本:1.0
#   作   者：崔宗会
#   完成日期：2019-x-x
# -----------------------------
import numpy as np
import matplotlib.pyplot as plt


def fun2ploy(x, n):
    """
    数据转化为[x^0,x^1,x^2,...x^n]
    :param x:
    :param n:
    :return:
    """
    len_num = len(x)
    X = np.ones([1, len_num])
    for i in range(1, n):
        Xt = np.vstack((X, np.power(x, i)))
        X = X.transpose()

    return X


def leastseq_byploy(x,y,ploy_dim):
    plt.scatter(x,y,edgecolors='r',marker='o',s=50)
    X=fun2ploy(x,ploy_dim)
    Xt=X.transpose()
    XtX=Xt.dot(X)
    XtXInv=np.linalg.inv(XtX)
    XtXInvXt=XtXInv.dot(Xt)
    coef=XtXInvXt.dot(y)
    y_est=X.dot(coef)
    return y_est,coef





def fit_fun(x):
    '''
    要拟合的函数
    '''
    return np.power(x,5)
    # return np.sin(x)


if __name__ == '__main__':
    data_num = 100
    ploy_dim = 6 # 拟合参数个数，即权重数量
    noise_scale = 0.2  # 数据准备
    x = np.array(np.linspace(-2 * np.pi, 2 * np.pi, data_num))  # 数据
    y = fit_fun(x) + noise_scale * np.random.rand(1, data_num)  # 添加噪声
    #转成列向量
    y=y.T
    color_list=['b','g',]


    [y_est,coef]=leastseq_byploy(x,y,ploy_dim)

    org_data=plt.scatter(x,y,edgecolors='r',marker='o',s=50)
    est_data=plt.plot(x,y_est,color='g',linewidth=3)

    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Fit function with leastseq method')
    plt.legend(['Noise data','Fited function'])
    plt.show()
