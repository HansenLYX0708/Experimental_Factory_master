import os
from aip import AipSpeech
from playsound import playsound
import Play_mp3

APPID = "25247462"
APPKEY = "nclKXKeecgdtmSzeewFI3S9M"
SECRETKEY = "d248emG3fOVwxBPjNY2iGKykvkgCYkPv"

client = AipSpeech(APPID, APPKEY, SECRETKEY)

str = '早上好'

result = client.synthesis(str,'zh', 1, {'vol':5, 'per':3})

'''
if not isinstance(result, dict):
    with open('auido.mp3', 'wb') as f:
        f.write(result)
'''


#Play_mp3.play('auido.mp3')

os.popen('ffplay -nodisp -i auido.mp3')

