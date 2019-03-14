# ----------------------------
#!  Copyright(C) 2019
#   All right reserved.
#   文件名称：xxx.py
#   摘   要：xxx
#   当前版本:1.0
#   作   者：崔宗会
#   完成日期：2019-x-x
# -----------------------------

from wxpy import *
import requests
import json


def auto_ai(text):
    url = "http://www.tuling123.com/openapi/api"
    api_key = "*******************" ####在这个网站http://www.turingapi.com/   注册一个机器人，获取api_key,填在这里
    payload = { "key": api_key,
                "info": text,
                "userid": "小明" }
    r = requests.post(url, data=json.dumps(payload))
    result = json.loads(r.content)
    return "[23世纪机器人]  " + result["text"]
bot = Bot(cache_path="./wxpy.pkl")
print('23世纪机器人') # 我的小号，测试需谨慎
list_friend=bot.friends()
print(list_friend)
my_friednd = bot.friends().search('小明')[0] # 如果想对所有好友实现机器人回复把参数my_friend改成chats = [Friend]
# my_friednd=bot.file_helper
@bot.register(my_friednd)
def my_friednd_message(msg):
    print('[接收]' + str(msg))
    if msg.type != 'Text':
        ret = '你给我看了什么！[拜托]'
    else:
        ret = auto_ai(msg.text)
        print('[发送]' + str(ret))
        return ret # 进入交互式的 Python 命令行界面，并堵塞当前线程
embed()


