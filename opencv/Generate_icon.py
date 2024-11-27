from PIL import Image

def make_ico_file(src, dist, size_list=None):
    def_size_list =[
        (256,256),
        (128, 128),
        (64, 64),
        (48, 48),
        (32, 32),
        (24, 24)
    ]
    my_size_list = size_list or def_size_list
    img = Image.open(src)
    img_crop = img.crop((0,0,img.size[0],img.size[1]))
    img_crop.save(dist, sizes=my_size_list)

if __name__ == '__main__':
    make_ico_file("3(已去底).png", "3.ico")