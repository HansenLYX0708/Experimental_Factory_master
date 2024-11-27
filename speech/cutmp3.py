from pydub import AudioSegment

# sound = AudioSegment.from_mp3("./mp3/haha.mp3")
# sound.export("./wav/haha.wav", format ='wav')

from scipy.io import wavfile

music = wavfile.read(r"./wav/mq53-好友革命（降噪＋音量调节.wav")
wavfile.write('./wav/mq53-好友革命（降噪＋音量调节-1分59.wav', 44100, music[1][0*44100:119*44100])   # 裁剪并保存音乐
