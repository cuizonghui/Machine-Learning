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

import tensorflow as tf
print(tf.__version__)

a=tf.constant(1)
with tf.Session() as sess:
    print(sess.run(a))



#
# import sys
# print(sys.version)
# # import pymysql
#
#
#
#
#
# # # 创建连接
# conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='root', db='database_name', charset='utf8')
# # 创建游标y
# cursor = conn.cursor()
#
# # 执行SQL，并返回收影响行数
# effect_row = cursor.execute("select * from tabe_name")
#
# # 执行SQL，并返回受影响行数
# #effect_row = cursor.execute("update tb7 set pass = '123' where nid = %s", (11,))
#
# # 执行SQL，并返回受影响行数,执行多次
# #effect_row = cursor.executemany("insert into tb7(user,pass,licnese)values(%s,%s,%s)", [("u1","u1pass","11111"),("u2","u2pass","22222")])
#
#
# # 提交，不然无法保存新建或者修改的数据
# conn.commit()
#
# # 关闭游标
# cursor.close()
# # 关闭连接
# conn.close()
