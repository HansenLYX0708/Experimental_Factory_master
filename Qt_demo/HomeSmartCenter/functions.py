import requests
import os
from aip import AipSpeech

def GetWeatherInfo():
    APPID = '32699543'
    APPSECRET = '78O1EqY0'
    city = '深圳'
    res = requests.get(
        'http://www.tianqiapi.com/api?version=v6&appid=' + APPID + "&appsecret=" + APPSECRET + '&city=' + city)
    res.encoding = 'utf-8'
    res_json = res.json()
    date = res_json['date']
    week = res_json['week']
    wea = res_json['wea']
    tip = res_json['air_tips']
    return '今天是'+date+week +','+wea

def GetAudioFromBaiduApi(str):
    APPID = "25247462"
    APPKEY = "nclKXKeecgdtmSzeewFI3S9M"
    SECRETKEY = "d248emG3fOVwxBPjNY2iGKykvkgCYkPv"
    client = AipSpeech(APPID, APPKEY, SECRETKEY)
    result = client.synthesis(str, 'zh', 1, {'vol': 5, 'per': 0})

    if not isinstance(result, dict):
        with open('auido.mp3', 'wb') as f:
            f.write(result)

    os.popen('ffplay -nodisp -i auido.mp3')