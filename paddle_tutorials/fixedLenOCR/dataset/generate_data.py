from captcha.image import ImageCaptcha
from random import randint

def gen_captcha(num, captcha_len):
    list = [chr(i) for i in range(48, 58)] + [chr(i) for i in range(65, 91)]

    for j in range(num):
        if j % 100 == 0:
            print(j)
        chars = ''
        for i in range(captcha_len):
            rand_num = randint(0, 35)
            chars += list[rand_num]
        image = ImageCaptcha(width=135, height=50, font_sizes=(35, 35, 35)).generate_image(chars)
        image.save('./train/' + chars + '.jpg')

if __name__ == '__main__':
    num = 500
    captcha_len = 5
    gen_captcha(num, captcha_len)