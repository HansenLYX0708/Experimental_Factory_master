import requests
import itchat
from itchat.content import *
import urllib
import random
import re
import os
import time


def get_response(msg):
    url = ''  # 看到请求url好像涉及到一些sessionid、userid等信息，可能直接复制会用不了什么的，所以你们直接去分析一下网页即可拿到啦，把content参数format成msg即可
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.79 Safari/537.36'}
    r = requests.get(url, headers=headers)
    response = re.findall('"body":{"fontStyle":0,"fontColor":0,"content":"(.*?)","emoticons":{}}}', r.text)[1].replace(
        '\\r\\n', '')
    return response


def get_words():
    words = []
    if os.path.exists('./words.txt'):
        with open('./words.txt', 'r', encoding='utf-8') as f:
            for i in f.read().split('\n')[:-1]:
                words.append(i)
    return words


def get_friendname():
    friends_name = {}  # 存储好友的微信昵称和备注
    friends = itchat.get_friends(update=True)  # 返回的是一个存储所有好友信息的列表，每个好友的信息都是用一个字典来存放
    for friend in friends[1:]:  # 第一个为自己，所以这里排除了自己
        friends_name.update({friend['UserName']: {'nickname': friend['NickName'], 'remarkname': friend['RemarkName']}})
    return friends_name


def get_username():
    chatrooms = itchat.get_chatrooms(update=True)  # 返回的是一个所有群聊的信息的列表，每个群聊信息都是用一个字典来存放
    user_name = []  # 接收特定群聊@本人的消息，并回复；存放特定群聊的username
    all_user_name = []  # 存放全部群聊的username
    vip = []  # 存放特定群聊的名称
    if os.path.exists('./vip.txt'):
        with open('./vip.txt', 'r', encoding='utf-8') as f:
            for i in f.read().split('\n')[:-1]:
                vip.append(i)
    for chatroom in chatrooms:
        all_user_name.append(chatroom['UserName'])
        if chatroom['NickName'] in vip:
            user_name.append(chatroom['UserName'])
    return all_user_name, user_name, vip


# 包括文本(表情符号)、位置、名片、通知、分享、图片(表情包)、语音、文件、视频
@itchat.msg_register([TEXT, MAP, CARD, SHARING, PICTURE, RECORDING, ATTACHMENT, VIDEO], isFriendChat=True,
                     isGroupChat=True)  # 监听个人消息和群聊消息
def download_reply_msg(msg):
    global flag, sj, isrun, use_info  # flag判断要不要进入斗图模式，sj控制斗图的时间长短，isrun判断是否启动自动回复机器人(默认运行中)，通过向传输助手发指令来控制，use_info说明文档
    all_user_name, user_name, vip = get_username()  # 每次接受消息时要拿到当前规定的群聊和特定群聊信息，后面用来分别做处理
    words = get_words()  # 拿到当前自定义回复消息的信息
    now_time = int(time.time())  # 记录获取这条消息的时间，后面处理撤回消息的时候用到
    b = []  # 用来记录已经过了可以撤回的时间的消息
    if len(msg_dict) != 0:
        for key, value in msg_dict.items():
            if (now_time - value['time']) >= 125:  # 经过验证发现消息2分钟之内才能撤回，这里为了保险起见加多5秒钟
                b.append(key)
        for eachkey in list(msg_dict.keys()):
            if eachkey in b:  # 要是过了撤回时间的消息是文件类型的就把它们删除，避免增加不必要的磁盘空间，盘大的请随意
                if 'file' in msg_dict[eachkey].keys():
                    os.remove(msg_dict[eachkey]['file'])
                msg_dict.pop(eachkey)
    # ---------------------------------------------------------
    # 下面开始存储各类消息，主要是用来查看别人撤回的消息，后面会用到
    if msg['Type'] in [MAP, SHARING]:  # 地图或者分享
        old_id = msg['MsgId']
        link = msg['Url']
        msg_dict.update({old_id: {'type': msg['Type'], 'data': link, 'time': now_time}})
    elif msg['Type'] in [PICTURE, RECORDING, ATTACHMENT, VIDEO]:
        if msg['ToUserName'] != 'filehelper':  # 避免给文件传输助手发文件也传入字典，没必要而且传入字典只是为了防止撤回，况且它是没有撤回的
            old_id = msg['MsgId']
            file = './保存的文件/' + msg['MsgId'] + '.' + msg['FileName'].split('.')[-1]
            msg['Text'](file)
            msg_dict.update({old_id: {'type': msg['Type'], 'file': file, 'time': now_time}})
        else:
            file = './保存的文件/' + msg['FileName']
            msg['Text'](file)
    elif msg['Type'] == CARD:  # 名片
        old_id = msg['MsgId']
        link = re.findall('bigheadimgurl="(.*)" smallheadimgurl', str(msg))[0]
        msg_content = '来自' + msg['RecommendInfo']['Province'] + msg['RecommendInfo']['City'] + '的' + \
                      msg['RecommendInfo']['NickName'] + '的名片'  # 内容就是推荐人的昵称和性别
        if msg['RecommendInfo']['Sex'] == 1:
            msg_content += '，男的'
        else:
            msg_content += '，女的'
        msg_dict.update({old_id: {'type': msg['Type'], 'head': link, 'data': msg_content, 'time': now_time}})
    elif msg['Type'] == TEXT:  # 文本
        old_id = msg['MsgId']
        text = msg['Text']
        msg_dict.update({old_id: {'type': msg['Type'], 'data': text, 'time': now_time}})
    # ---------------------------------------------------------
    # 下面是自动回复消息的（一切回复逻辑都在这里）
    if msg['ToUserName'] != 'filehelper':  # 避免给文件传输助手发消息也自动回复
        if isrun == '运行中......':  # 操控机器人的，想停就停，想启动就启动，不用关掉程序，而且不影响查看撤回消息的功能
            if msg['FromUserName'] in all_user_name:
                if msg['FromUserName'] in user_name:  # 当消息来自特定群聊时，下面代码才会执行
                    if sj is not None:
                        if int(time.time()) - sj >= 900:  # 斗图时间：15分钟
                            flag = 0
                            sj = None
                    if (msg['isAt'] is True) & (msg['Type'] == TEXT):
                        myname = '@' + re.findall("'Self'.*?'DisplayName': '(.*?)', 'KeyWord'", str(msg))[0] if \
                        re.findall("'Self'.*?'DisplayName': '(.*?)', 'KeyWord'", str(msg))[0] != '' else '这里填你自己的微信昵称'
                        if '帅哥来斗图' in msg['Text']:
                            flag = 1
                            sj = int(time.time())
                            num = random.choice(os.listdir('./表情包'))
                            msg.user.send('@img@./表情包/{}'.format(num))
                            return None
                        reply = get_response(msg['Text'].replace(myname, ''))
                        if 'I am' in reply:
                            reply = reply.replace('小i机器人', 'your father')
                        if '小i' in reply:  # 这个我是不想让别人知道是小i机器人，才把它换掉的，你想换成什么随你，不换也行，就把这段代码删除即可
                            reply = reply.replace('小i', '你爸爸')
                        if '机器人' in reply:
                            reply = reply.replace('机器人', '')
                        if '输入' in reply:
                            if flag == 0:
                                reply = '有种来斗图，输入“帅哥来斗图”即可。'
                            else:
                                reply = random.choice(words)
                        itchat.send('@%s\u2005%s' % (msg['ActualNickName'], reply), msg['FromUserName'])
                    if (msg['Type'] == PICTURE) & (flag == 1):
                        num = random.choice(os.listdir('./表情包'))
                        msg.user.send('@img@./表情包/{}'.format(num))
                else:  # 这里是当消息来自不是特定群聊时，要执行的代码
                    if msg['Type'] == TEXT:
                        if '收到请回复' in msg['Text']:
                            return '收到'
            else:  # 下面是处理个人消息的
                # 经过测试发现如果自己手动发消息到新建的群中，也会触发下面自动回复的代码，于是就要排除这个bug用下面第一个if语句
                if msg['FromUserName'] == itchat.search_friends(nickName='这里填你自己的微信昵称')[0]['UserName']:
                    return None
                if msg['Type'] == TEXT:  # 下面跟处理群聊的时候差不多，就不重复了嘻嘻
                    reply = get_response(msg['Text'])
                    if 'I am' in reply:
                        reply = reply.replace('小i机器人', 'your father')
                    if '小i' in reply:
                        reply = reply.replace('小i', '你爸爸')
                    if '机器人' in reply:
                        reply = reply.replace('机器人', '')
                    if '输入' in reply:
                        reply = random.choice(words)
                    msg.user.send(reply + '\n                                    [不是本人]')  # 36个空格
                elif msg['Type'] == PICTURE:  # 表情包回复
                    num = random.choice(os.listdir('./表情包'))
                    msg.user.send('@img@./表情包/{}'.format(num))
                elif msg['Type'] == RECORDING:
                    msg.user.send('请打字和我交流，谢谢。' + '\n                                    [不是本人]')
    # ---------------------------------------------------------
    # 下面是用来控制机器人的（给文件传输助手发指令）代码很简单，也很清晰
    else:
        if msg['Type'] == TEXT:
            if '添加vip' in msg['Text']:
                with open('./vip.txt', 'a', encoding='utf-8') as f:
                    f.write(msg['Text'][5:])
                    f.write('\n')
            if '查看vip' in msg['Text']:
                now_vip = '\n'.join(vip)
                itchat.send('当前的vip群有：\n{0}'.format(now_vip), toUserName='filehelper')
            if '删除vip' in msg['Text']:
                if os.path.exists('./vip.txt'):
                    with open('./vip.txt', 'r', encoding='utf-8') as f1:
                        lines = f1.readlines()
                        with open('./vip.txt', 'w', encoding='utf-8') as f2:
                            for line in lines:
                                if msg['Text'][5:] != line.strip():
                                    f2.write(line)
            if '清空vip' in msg['Text']:
                with open('./vip.txt', 'w', encoding='utf-8') as f:
                    f.flush()

            if '添加words' in msg['Text']:
                with open('./words.txt', 'a', encoding='utf-8') as f:
                    f.write(msg['Text'][7:])
                    f.write('\n')
            if '查看words' in msg['Text']:
                now_words = '\n'.join(words)
                itchat.send('当前的words有：\n{0}'.format(now_words), toUserName='filehelper')
            if '删除words' in msg['Text']:
                if os.path.exists('./words.txt'):
                    with open('./words.txt', 'r', encoding='utf-8') as f1:
                        lines = f1.readlines()
                        with open('./words.txt', 'w', encoding='utf-8') as f2:
                            for line in lines:
                                if msg['Text'][7:] != line.strip():
                                    f2.write(line)
            if '清空words' in msg['Text']:
                with open('./words.txt', 'w', encoding='utf-8') as f:
                    f.flush()
            if '停止机器人' in msg['Text']:
                isrun = '已停止！'
            if '启动机器人' in msg['Text']:
                isrun = '运行中......'
            if '查看机器人' in msg['Text']:
                itchat.send(isrun, toUserName='filehelper')
            if 'robot' in msg['Text']:
                itchat.send(use_info, toUserName='filehelper')


@itchat.msg_register(NOTE, isFriendChat=True, isGroupChat=True)
def get_note(msg):
    if '撤回了一条消息' in msg['Text']:
        if '你撤回了一条消息' in msg['Text']:
            return None
        new_id = re.findall('<msgid>(\d+)</msgid>', str(msg))[0]
        public_time = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(msg['CreateTime'] + 28800))
        data = msg_dict[new_id]
        nickname = re.findall("'UserName': '@[a-zA-Z0-9]+', 'NickName': '(.*?)', 'HeadImgUrl'", str(msg))[0]
        if len(nickname) > 100:
            friends_name = get_friendname()
            # 群名这句我觉得还会有bug，由于测试的时候只在2个人的群中测试，多个的话可能会出bug，这个后面再说吧，反正我24小时挂着，有bug咱再说哈哈
            qun = re.findall("'NickName': '.*",
                             re.findall("'UserName': '@[a-zA-Z0-9]+', 'NickName': '(.*)', 'HeadImgUrl'", str(msg))[0][
                             -100:])[0].replace("'NickName': '", '')
            an = msg['ActualNickName']
            nickname = '{0}在{1}群中'.format(an, qun)
            if msg['ActualUserName'] in friends_name.keys():
                if friends_name[msg['ActualUserName']]['remarkname'] != '':
                    nickname = '{0}({1})在{2}群中'.format(an, friends_name[msg['ActualUserName']]['remarkname'], qun)
                else:
                    nickname = '{0}({1})在{2}群中'.format(an, friends_name[msg['ActualUserName']]['nickname'], qun)
        if data['type'] == MAP:
            itchat.send('%s在%s撤回了一个位置，其位置：\n%s' % (nickname, public_time, data['data']), toUserName='filehelper')
        elif data['type'] == CARD:
            itchat.send('%s在%s撤回了一个名片，名片信息：\n%s，其头像：' % (nickname, public_time, data['data']), toUserName='filehelper')
            itchat.send('%s' % (data['head']), toUserName='filehelper')
        elif data['type'] == SHARING:
            itchat.send('%s在%s撤回了一个分享，其链接：\n%s' % (nickname, public_time, data['data']), toUserName='filehelper')
        elif data['type'] == TEXT:
            itchat.send('%s在%s撤回了一个信息，其内容：\n%s' % (nickname, public_time, data['data']), toUserName='filehelper')
        elif data['type'] in [PICTURE, RECORDING, ATTACHMENT, VIDEO]:
            itchat.send('%s在%s撤回了一张图片或者一个表情或者一段语音或者一个视频又或者一个文件，如下：' % (nickname, public_time), toUserName='filehelper')
            itchat.send('@%s@%s' % ({'Picture': 'img', 'Video': 'vid'}.get(data['type'], 'fil'), data['file']),
                        toUserName='filehelper')


# 下面开始运行机器人和创建所需目录以及定义默认变量
itchat.auto_login(hotReload=True) #windows下用这个
# itchat.auto_login(hotReload=True, enableCmdQR=2)  # linux下用这个
msg_dict = {}  # 储存聊天记录，找到撤回的消息
flag = 0  # 判断vip群要不要进入斗图模式（默认不会，flag为1时就会）
sj = None  # 判断啥时候flag恢复为0
use_info = '一、特殊群聊管理\n①增加：添加vip+群名\n②删除：删除vip+群名\n③查询：查看vip+群名\n④清空：清空vip+群名\n二、默认回复管理\n①增加：添加words+语句\n②删除：删除words+语句\n③查询：查看words+语句\n④清空：清空words+语句\n三、robot管理\n①启动：启动机器人\n②停止：停止机器人\n③状态：查看机器人\n四、说明文档：\n①指令：robot\n②注意：如果添加的群没有名字，那么把群里每个人的昵称用英文逗号隔开'  # 说明文档
isrun = '运行中......'  # 是否启动自动回复机器人(默认运行中)，通过向传输助手发指令来控制
if not os.path.exists('./表情包'):
    os.makedirs('./表情包')
if not os.path.exists('./保存的文件'):
    os.makedirs('./保存的文件')
itchat.run()