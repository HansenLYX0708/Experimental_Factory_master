import requests

r = requests.get('http://www.weather.com.cn/data/sk/101280601.html')  # 101280601 shen zhen
r.encoding = 'utf-8'
res = r.json()
print(r.json()['weatherinfo']['city'], r.json()['weatherinfo']['WD'], r.json()['weatherinfo']['temp'])