# ----------------------------
#!  Copyright(C) 2019
#   All right reserved.
#   文件名称：xxx.py
#   摘   要：xxx
#   当前版本:1.0
#   作   者：崔宗会
#   完成日期：2019-x-x
# -----------------------------
import matplotlib.pyplot as plt
import numpy as np


def cost_function(w):
    return w**2 - 8 * w + 4


def gradient_function(w):
    return (2 * w - 8)


def gradient_descent(w, learning_rate):
    gradient_value = gradient_function(w)
    w = w - learning_rate * gradient_value
    return w


def momentum_gradient_descent(w, v, learning_rate, beta):
    gradient_value = gradient_function(w)
    v = beta * v + (1 - beta) * gradient_value
    w = w - learning_rate * v
    return w, v


def nesterov_accelerated_gradient(w, v, learning_rate, beta):
    gradient_value = gradient_function(w - beta * v)
    v = beta * v + (1 - beta) * gradient_value
    w = w - learning_rate * v
    return w, v


def adaptive_gradient(w, g2, learning_rate):
    gradient_value = gradient_function(w)
    g2 += gradient_value ** 2
    w = w - (learning_rate / (np.sqrt(g2 + 1e-8))) * gradient_value
    return w, g2


def RMSprop(w, Eg2, learning_rate, beta):
    gradient_value = gradient_function(w)
    Eg2 = beta * Eg2 + (1 - beta) * gradient_value * gradient_value
    w = w - (learning_rate / (np.sqrt(Eg2 + 1e-8))) * gradient_value
    return w, Eg2


def Adadelta(w, Eg2, Edeta_w2, beta):
    gradient_value = gradient_function(w)
    Eg2 = beta * Eg2 + (1 - beta) * gradient_value * gradient_value
    rms_gt = np.sqrt(Eg2 + 1e-6)
    rms_deta_w = np.sqrt(Edeta_w2 + 1e-5)
    deta_w = -rms_deta_w / (rms_gt) * gradient_value
    Edeta_w2 = beta * Edeta_w2 + (1 - beta) * deta_w * deta_w
    w = w + deta_w
    return w, Eg2, Edeta_w2


def Adam(w, v, g2, learning_rate, beat1, beta2, t):
    gradient_value = gradient_function(w)
    v = beat1 * v + (1 - beat1) * gradient_value
    g2 = beta2 * g2 + (1 - beta2) * gradient_value * gradient_value
    v_hat = v / (1 - np.power(beat1, t))
    g2_hat = g2 / (1 - np.power(beta2, t))
    t += 1
    w = w - learning_rate * v_hat / (np.sqrt(g2_hat) + 1e-8)
    return w, v, g2, t


def main():

    epoch = 500
    learning_rate = 0.01
    beta = 0.9
    w_gd = 0
    w_gd_list = []
    cost_gd_list = []

    v_mgd = 0
    w_mgd = 0
    w_mgd_list = []
    cost_mgd_list = []

    v_nag = 0
    w_nag = 0
    w_nag_list = []
    cost_nag_list = []

    g_adagrad = 0
    w_adagrad = 0
    w_adagrad_list = []
    cost_adagrad_list = []

    Eg_rmsprop = 0
    w_rmsprop = 0
    w_rmsprop_list = []
    cost_rmsprop_list = []

    Eg_adadelta = 0
    Edeta_w_adadelta = 0
    w_adadelta_list = []
    cost_adadelta_list = []
    w_adadelta = 0

    Eg_adam = 0
    w_adam = 0
    w_adam_list = []
    cost_adam_list = []
    v_adam = 0
    beta2 = 0.9999
    t = 1

    for i in range(epoch):
        # gd
        w_gd_list.append(w_gd)
        cost_gd_list.append(cost_function(w_gd))
        w_gd = gradient_descent(w_gd, learning_rate)
        # print(gradient_function(w_gd))
        # mgd
        w_mgd_list.append(w_mgd)
        cost_mgd_list.append(cost_function(w_mgd))
        w_mgd, v_mgd = momentum_gradient_descent(w_mgd, v_mgd, learning_rate, beta)

        # nag
        w_nag_list.append(w_nag)
        cost_nag_list.append(cost_function(w_nag))
        w_nag, v_nag = momentum_gradient_descent(w_nag, v_nag, learning_rate, beta)
        # agagrad
        w_adagrad_list.append(w_adagrad)
        cost_adagrad_list.append(cost_function(w_adagrad))
        w_adagrad, g_adagrad = adaptive_gradient(w_adagrad, g_adagrad, learning_rate * 50)

        # rmgprop

        w_rmsprop_list.append(w_rmsprop)
        cost_rmsprop_list.append(cost_function(w_rmsprop))
        w_rmsprop, Eg_rmsprop = RMSprop(w_rmsprop, Eg_rmsprop, learning_rate * 50, beta)

        # adadelta
        w_adadelta_list.append(w_adadelta)
        cost_adadelta_list.append(cost_function(w_adadelta))
        w_adadelta, Eg_adadelta, Edeta_w_adadelta = Adadelta(w_adadelta, Eg_adadelta, Edeta_w_adadelta, beta=0.9)

        w_adam_list.append(w_adam)
        cost_adam_list.append(cost_function(w_adam))
        w_adam, v_adam, Eg_adam, t = Adam(w_adam, v_adam, Eg_adam, 50 * learning_rate, beta, beta2, t)

    plt.show()


if __name__ == '__main__':
    main()
