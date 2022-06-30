import requests

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

if __name__ == '__main__':
    str = GetWeatherInfo()
    print(str)


